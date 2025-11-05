"""
多様性を考慮した再ランキング

MMR (Maximal Marginal Relevance)、カテゴリ多様性、タイプ多様性などを
考慮して推薦結果を再ランキングする

改善内容:
1. MMRの効率化（キャッシング、早期終了）
2. Position-aware ranking（上位は精度、下位は多様性）
3. より柔軟な多様性戦略
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class DiversityReranker:
    """多様性を考慮した再ランキングクラス"""

    def __init__(self,
                 lambda_relevance: float = 0.7,
                 category_weight: float = 0.5,
                 type_weight: float = 0.3):
        """
        初期化

        Args:
            lambda_relevance: 関連性の重み（0-1）、高いほど精度重視、低いほど多様性重視
            category_weight: カテゴリ多様性の重み（0-1）
            type_weight: タイプ多様性の重み（0-1）
        """
        self.lambda_relevance = lambda_relevance
        self.category_weight = category_weight
        self.type_weight = type_weight

    def rerank_mmr(self,
                   candidates: List[Tuple[str, float]],
                   competence_info: pd.DataFrame,
                   k: int = 10,
                   use_position_aware: bool = False) -> List[Tuple[str, float]]:
        """
        MMR (Maximal Marginal Relevance)による再ランキング（効率化版）

        改善点:
        - 類似度計算のキャッシング
        - Position-aware ranking: 上位は精度重視、下位は多様性重視
        - 早期終了による高速化

        Args:
            candidates: (力量コード, スコア)のリスト（スコア降順でソート済み想定）
            competence_info: 力量情報DataFrame（力量コード, 力量カテゴリー名, 力量タイプ等）
            k: 最終的な推薦数
            use_position_aware: position-aware rankingを使用するか

        Returns:
            再ランキングされた(力量コード, スコア)のリスト
        """
        if len(candidates) == 0:
            return []

        # 力量情報をマッピング
        competence_dict = competence_info.set_index('力量コード').to_dict('index')

        # 類似度のキャッシュ
        similarity_cache = {}

        def cached_similarity(code1: str, code2: str) -> float:
            """キャッシュを使った類似度計算"""
            key = tuple(sorted([code1, code2]))
            if key not in similarity_cache:
                similarity_cache[key] = self._calculate_similarity(
                    competence_dict.get(code1, {}),
                    competence_dict.get(code2, {})
                )
            return similarity_cache[key]

        selected = []
        remaining = list(candidates)
        max_score = candidates[0][1] if candidates else 1.0  # 正規化用

        while len(selected) < k and remaining:
            best_idx = -1
            best_mmr_score = -np.inf

            # Position-aware ranking: 位置によってlambdaを動的に調整
            if use_position_aware:
                # 上位は精度重視（lambda高）、下位は多様性重視（lambda低）
                position_ratio = len(selected) / k
                current_lambda = self.lambda_relevance * (1 - 0.3 * position_ratio)
            else:
                current_lambda = self.lambda_relevance

            for idx, (comp_code, relevance_score) in enumerate(remaining):
                # 関連性スコア（正規化）
                rel_score = relevance_score / max_score if max_score > 0 else 0

                # 多様性スコア：選択済みアイテムとの非類似度
                if len(selected) == 0:
                    diversity_score = 1.0  # 最初のアイテムは多様性最大
                else:
                    # 選択済みアイテムとの最大類似度（キャッシュ使用）
                    max_similarity = max(
                        cached_similarity(comp_code, sel_code)
                        for sel_code, _ in selected
                    )
                    diversity_score = 1.0 - max_similarity

                # MMRスコア（position-aware lambdaを使用）
                mmr_score = (current_lambda * rel_score +
                            (1 - current_lambda) * diversity_score)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break  # 早期終了

        return selected

    def rerank_category_diversity(self,
                                  candidates: List[Tuple[str, float]],
                                  competence_info: pd.DataFrame,
                                  k: int = 10,
                                  max_per_category: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        カテゴリ多様性を考慮した再ランキング

        Args:
            candidates: (力量コード, スコア)のリスト
            competence_info: 力量情報DataFrame
            k: 最終的な推薦数
            max_per_category: カテゴリごとの最大推薦数（Noneの場合は制限なし）

        Returns:
            カテゴリ多様性を考慮した(力量コード, スコア)のリスト
        """
        if len(candidates) == 0:
            return []

        # 力量情報をマッピング
        competence_dict = competence_info.set_index('力量コード').to_dict('index')

        # カテゴリごとのカウント
        category_counts = defaultdict(int)
        selected = []

        for comp_code, score in candidates:
            if len(selected) >= k:
                break

            # カテゴリを取得
            comp_info = competence_dict.get(comp_code, {})
            category = comp_info.get('力量カテゴリー名', 'Unknown')

            # カテゴリごとの制限をチェック
            if max_per_category is None or category_counts[category] < max_per_category:
                selected.append((comp_code, score))
                category_counts[category] += 1

        # 制限により不足した場合、残りを追加
        if len(selected) < k:
            for comp_code, score in candidates:
                if len(selected) >= k:
                    break
                if (comp_code, score) not in selected:
                    selected.append((comp_code, score))

        return selected[:k]

    def rerank_type_diversity(self,
                              candidates: List[Tuple[str, float]],
                              competence_info: pd.DataFrame,
                              k: int = 10,
                              type_ratios: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        タイプ多様性を考慮した再ランキング（SKILL/EDUCATION/LICENSE）

        Args:
            candidates: (力量コード, スコア)のリスト
            competence_info: 力量情報DataFrame
            k: 最終的な推薦数
            type_ratios: タイプごとの目標比率（例: {'SKILL': 0.6, 'EDUCATION': 0.3, 'LICENSE': 0.1}）

        Returns:
            タイプ多様性を考慮した(力量コード, スコア)のリスト
        """
        if len(candidates) == 0:
            return []

        # デフォルトの比率
        if type_ratios is None:
            type_ratios = {'SKILL': 0.5, 'EDUCATION': 0.3, 'LICENSE': 0.2}

        # 力量情報をマッピング
        competence_dict = competence_info.set_index('力量コード').to_dict('index')

        # 候補をタイプごとに分類
        candidates_by_type = defaultdict(list)
        for comp_code, score in candidates:
            comp_info = competence_dict.get(comp_code, {})
            comp_type = comp_info.get('力量タイプ', 'SKILL')
            candidates_by_type[comp_type].append((comp_code, score))

        # 各タイプから目標比率に従って選択
        selected = []
        type_targets = {t: int(k * ratio) for t, ratio in type_ratios.items()}

        # まず目標数まで各タイプから選択
        for comp_type, target_count in type_targets.items():
            type_candidates = candidates_by_type.get(comp_type, [])
            selected.extend(type_candidates[:target_count])

        # 不足分を残りから補充（スコア順）
        if len(selected) < k:
            remaining = [
                (comp_code, score)
                for comp_code, score in candidates
                if (comp_code, score) not in selected
            ]
            selected.extend(remaining[:k - len(selected)])

        # スコアでソート
        selected.sort(key=lambda x: x[1], reverse=True)

        return selected[:k]

    def rerank_hybrid(self,
                     candidates: List[Tuple[str, float]],
                     competence_info: pd.DataFrame,
                     k: int = 10,
                     max_per_category: Optional[int] = 3,
                     type_ratios: Optional[Dict[str, float]] = None,
                     use_position_aware: bool = True) -> List[Tuple[str, float]]:
        """
        ハイブリッド再ランキング（MMR + カテゴリ多様性 + タイプ多様性）

        改善点:
        - Position-aware rankingのサポート
        - より効率的なパイプライン

        Args:
            candidates: (力量コード, スコア)のリスト
            competence_info: 力量情報DataFrame
            k: 最終的な推薦数
            max_per_category: カテゴリごとの最大推薦数
            type_ratios: タイプごとの目標比率
            use_position_aware: position-aware rankingを使用するか

        Returns:
            ハイブリッド再ランキングされた(力量コード, スコア)のリスト
        """
        # Step 1: MMRで多様性を考慮したTop候補を選択（position-aware対応）
        mmr_candidates = self.rerank_mmr(
            candidates, competence_info, k=k * 2, use_position_aware=use_position_aware
        )

        # Step 2: カテゴリ多様性を適用
        category_diverse = self.rerank_category_diversity(
            mmr_candidates, competence_info, k=k * 2, max_per_category=max_per_category
        )

        # Step 3: タイプ多様性を適用
        final_ranking = self.rerank_type_diversity(
            category_diverse, competence_info, k=k, type_ratios=type_ratios
        )

        return final_ranking

    def _calculate_similarity(self, comp1: Dict, comp2: Dict) -> float:
        """
        2つの力量の類似度を計算

        Args:
            comp1: 力量1の情報辞書
            comp2: 力量2の情報辞書

        Returns:
            類似度（0-1）
        """
        similarity = 0.0

        # カテゴリが同じ
        if comp1.get('力量カテゴリー名') == comp2.get('力量カテゴリー名'):
            similarity += self.category_weight

        # タイプが同じ
        if comp1.get('力量タイプ') == comp2.get('力量タイプ'):
            similarity += self.type_weight

        # 正規化
        max_similarity = self.category_weight + self.type_weight
        if max_similarity > 0:
            similarity = similarity / max_similarity

        return min(similarity, 1.0)

    def calculate_diversity_metrics(self,
                                    recommendations: List[Tuple[str, float]],
                                    competence_info: pd.DataFrame) -> Dict[str, float]:
        """
        推薦結果の多様性指標を計算

        Args:
            recommendations: (力量コード, スコア)のリスト
            competence_info: 力量情報DataFrame

        Returns:
            多様性指標の辞書
        """
        if len(recommendations) == 0:
            return {
                'category_diversity': 0.0,
                'type_diversity': 0.0,
                'intra_list_diversity': 0.0
            }

        competence_dict = competence_info.set_index('力量コード').to_dict('index')

        # カテゴリ多様性：ユニークなカテゴリ数 / 推薦数
        categories = set()
        types = set()
        for comp_code, _ in recommendations:
            comp_info = competence_dict.get(comp_code, {})
            categories.add(comp_info.get('力量カテゴリー名', 'Unknown'))
            types.add(comp_info.get('力量タイプ', 'SKILL'))

        category_diversity = len(categories) / len(recommendations)
        type_diversity = len(types) / len(recommendations)

        # Intra-List Diversity: アイテム間の平均非類似度
        similarities = []
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                comp1 = competence_dict.get(recommendations[i][0], {})
                comp2 = competence_dict.get(recommendations[j][0], {})
                sim = self._calculate_similarity(comp1, comp2)
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0.0
        intra_list_diversity = 1.0 - avg_similarity

        return {
            'category_diversity': category_diversity,
            'type_diversity': type_diversity,
            'intra_list_diversity': intra_list_diversity,
            'unique_categories': len(categories),
            'unique_types': len(types)
        }
