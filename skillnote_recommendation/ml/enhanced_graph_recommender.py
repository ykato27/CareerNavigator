"""
Enhanced Graph-based Recommender

改良されたグラフベース推薦モデル

主な改善点:
1. 時間減衰重み付け: 最近の遷移を重視
2. パス品質評価: 長さ・強度・新規性を総合評価
3. 効率的なパス探索: 重要度ベースのランキング
4. Robust Scaling: 外れ値に強い正規化
5. 詳細な統計情報の保存
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from datetime import datetime, timedelta
from scipy.stats import rankdata

from skillnote_recommendation.ml.base_recommender import BaseRecommender, Recommendation

logger = logging.getLogger(__name__)


class EnhancedSkillTransitionGraphRecommender(BaseRecommender):
    """
    改良版スキル遷移グラフベースの推薦モデル

    主な改善:
    - 時間減衰重み付け（指数減衰）
    - パス品質スコアリング
    - Robust Scaling による正規化
    - 動的重み調整
    """

    def __init__(
        self,
        time_window_days: int = 180,
        min_transition_count: int = 2,
        embedding_dim: int = 64,
        walk_length: int = 10,
        num_walks: int = 80,
        p: float = 1.0,
        q: float = 2.0,
        workers: int = 4,
        time_decay_factor: float = 0.01,  # 時間減衰係数（1日あたり）
        use_time_decay: bool = True,
        path_quality_weight: float = 0.3,  # パス品質の重み
        use_robust_scaling: bool = True
    ):
        """
        初期化

        Args:
            time_window_days: 遷移とみなす最大期間（日数）
            min_transition_count: 最小遷移人数
            embedding_dim: 埋め込み次元数
            walk_length: ランダムウォークの長さ
            num_walks: ウォーク回数
            p: Return parameter（DFS vs BFS）
            q: In-out parameter（local vs global）
            workers: 並列処理数
            time_decay_factor: 時間減衰係数（0.01 = 1日あたり1%減衰）
            use_time_decay: 時間減衰重み付けを使用
            path_quality_weight: パス品質スコアの重み
            use_robust_scaling: Robust Scalingを使用（外れ値に強い）
        """
        super().__init__(
            name="EnhancedSkillTransitionGraph",
            interpretability_score=5  # 改善により解釈性が向上
        )

        self.time_window_days = time_window_days
        self.min_transition_count = min_transition_count
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.time_decay_factor = time_decay_factor
        self.use_time_decay = use_time_decay
        self.path_quality_weight = path_quality_weight
        self.use_robust_scaling = use_robust_scaling

        self.graph = None
        self.node2vec_model = None
        self.transition_stats = {}
        self.edge_details = {}  # 詳細なエッジ情報

    def fit(self, member_competence: pd.DataFrame, competence_master: pd.DataFrame) -> None:
        """
        モデルの学習

        Args:
            member_competence: メンバー習得力量データ
            competence_master: 力量マスタデータ
        """
        logger.info("=" * 80)
        logger.info("EnhancedSkillTransitionGraphRecommender の学習を開始")
        logger.info("=" * 80)

        self.member_competence = member_competence.copy()
        self.competence_master = competence_master.copy()

        # 取得日カラムの確認
        if '取得日' not in member_competence.columns:
            raise ValueError("スキル遷移グラフには「取得日」カラムが必要です")

        # グラフ構築
        logger.info("\nStep 1: 改良版スキル遷移グラフの構築")
        self.graph = self._build_enhanced_transition_graph()
        logger.info(f"  ノード数: {self.graph.number_of_nodes()}")
        logger.info(f"  エッジ数: {self.graph.number_of_edges()}")
        logger.info(f"  時間減衰: {'有効' if self.use_time_decay else '無効'}")

        # Node2Vec学習
        if self.graph.number_of_nodes() > 1:
            logger.info("\nStep 2: Node2Vec埋め込みの学習")
            self._train_node2vec()
            logger.info(f"  埋め込み次元: {self.embedding_dim}")

        self.is_fitted = True
        self.metadata = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'time_window_days': self.time_window_days,
            'time_decay_enabled': self.use_time_decay,
            'has_embeddings': self.node2vec_model is not None
        }

        logger.info("\n" + "=" * 80)
        logger.info("学習完了")
        logger.info("=" * 80)

    def _build_enhanced_transition_graph(self) -> nx.DiGraph:
        """
        改良版スキル遷移グラフを構築

        改善点:
        - 時間減衰重み付け
        - 詳細な統計情報の保存
        - 効率的な計算（連続ペアのみ）

        Returns:
            有向グラフ
        """
        G = nx.DiGraph()

        # 取得日を日付型に変換
        df = self.member_competence.copy()
        df['取得日_dt'] = pd.to_datetime(df['取得日'], errors='coerce')
        df = df[df['取得日_dt'].notna()]

        # 現在日時（時間減衰の基準点）
        now = pd.Timestamp.now()

        # メンバーごとに学習順序を抽出
        transition_data = {}

        for member in df['メンバーコード'].unique():
            member_skills = df[
                df['メンバーコード'] == member
            ].sort_values('取得日_dt')

            skills = member_skills[['力量コード', '取得日_dt']].values

            # 連続するスキルペアのみを抽出（効率化）
            for i in range(len(skills) - 1):
                source_skill, source_date = skills[i]
                target_skill, target_date = skills[i + 1]

                # 時間窓内の遷移のみ
                time_diff = (target_date - source_date).days
                if time_diff <= self.time_window_days:
                    edge = (source_skill, target_skill)

                    if edge not in transition_data:
                        transition_data[edge] = {
                            'transitions': [],
                            'acquisition_dates': []
                        }

                    # 時間減衰重みを計算
                    days_ago = (now - target_date).days
                    if self.use_time_decay:
                        decay_weight = np.exp(-self.time_decay_factor * days_ago / 365)
                    else:
                        decay_weight = 1.0

                    transition_data[edge]['transitions'].append({
                        'time_diff': time_diff,
                        'decay_weight': decay_weight,
                        'acquisition_date': target_date
                    })
                    transition_data[edge]['acquisition_dates'].append(target_date)

        # エッジを追加
        for (source, target), data in transition_data.items():
            transitions = data['transitions']
            count = len(transitions)

            if count >= self.min_transition_count:
                # 時間減衰重み付きカウント
                weighted_count = sum(t['decay_weight'] for t in transitions)

                # 統計情報を計算
                time_diffs = [t['time_diff'] for t in transitions]
                avg_time_diff = np.mean(time_diffs)
                median_time_diff = np.median(time_diffs)
                std_time_diff = np.std(time_diffs)

                # 最新の遷移日
                latest_transition = max(data['acquisition_dates'])
                recency_days = (now - latest_transition).days

                # エッジを追加
                G.add_edge(
                    source,
                    target,
                    weight=weighted_count,
                    raw_count=count,
                    avg_time_diff=avg_time_diff,
                    median_time_diff=median_time_diff,
                    std_time_diff=std_time_diff,
                    recency_days=recency_days,
                )

                # 詳細情報を保存
                self.edge_details[(source, target)] = {
                    'count': count,
                    'weighted_count': weighted_count,
                    'avg_days': avg_time_diff,
                    'median_days': median_time_diff,
                    'std_days': std_time_diff,
                    'recency_days': recency_days,
                    'transitions': transitions
                }

        return G

    def _train_node2vec(self) -> None:
        """Node2Vecで埋め込みを学習"""
        try:
            node2vec = Node2Vec(
                self.graph,
                dimensions=self.embedding_dim,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                p=self.p,
                q=self.q,
                workers=self.workers,
                quiet=True
            )

            self.node2vec_model = node2vec.fit(
                window=5,
                min_count=1,
                batch_words=4,
                epochs=5
            )

        except Exception as e:
            logger.error(f"Node2Vec学習中にエラー: {e}")
            self.node2vec_model = None

    def recommend(
        self,
        member_code: str,
        n: int = 10,
        exclude_acquired: bool = True,
        competence_type: Optional[List[str]] = None
    ) -> List[Recommendation]:
        """
        推薦リストの生成（改良版）

        Args:
            member_code: メンバーコード
            n: 推薦する件数
            exclude_acquired: 既習得スキルを除外するか
            competence_type: フィルタする力量タイプのリスト（例: ['SKILL', 'EDUCATION']）
                             Noneの場合は全ての力量タイプを推薦

        Returns:
            推薦結果のリスト
        """
        self._check_fitted()

        user_skills = self.get_user_skills(member_code)

        if not user_skills:
            logger.warning(f"メンバー {member_code} は習得スキルがありません")
            return []

        # 候補スキルを収集
        candidates = {}

        # 方法1: グラフの隣接ノード（時間減衰重み付き）
        for skill in user_skills:
            if skill in self.graph:
                for neighbor in self.graph.neighbors(skill):
                    if exclude_acquired and neighbor in user_skills:
                        continue

                    edge_data = self.graph[skill][neighbor]
                    weighted_count = edge_data['weight']

                    # パス品質スコアを計算
                    path_quality = self._calculate_path_quality(
                        source=skill,
                        target=neighbor,
                        path_length=2
                    )

                    # 総合スコア
                    score = weighted_count * (1 + self.path_quality_weight * path_quality)

                    if neighbor not in candidates or candidates[neighbor]['score'] < score:
                        candidates[neighbor] = {
                            'score': score,
                            'source_skill': skill,
                            'reason_type': 'direct_transition',
                            'metadata': edge_data,
                            'path_quality': path_quality
                        }

        # 方法2: 埋め込み空間での類似度（Robust Scaling）
        if self.node2vec_model is not None:
            embedding_scores = self._get_embedding_recommendations(
                user_skills, exclude_acquired, n * 2
            )

            for skill, score in embedding_scores.items():
                if skill in candidates:
                    # 既存スコアとブレンド
                    candidates[skill]['score'] += score
                    candidates[skill]['has_embedding_similarity'] = True
                else:
                    candidates[skill] = {
                        'score': score,
                        'source_skill': user_skills[0],
                        'reason_type': 'embedding_similarity',
                        'metadata': {},
                        'path_quality': 0.5
                    }

        # 力量タイプでフィルタリング
        if competence_type is not None:
            logger.info(f"力量タイプフィルタ適用開始: {competence_type}")
            logger.info(f"フィルタ前の候補数: {len(candidates)}")

            filtered_candidates = {}
            rejected_count = 0
            for skill_code, data in candidates.items():
                skill_type = self._get_skill_type(skill_code)
                skill_name = self.get_skill_name(skill_code)

                if skill_type in competence_type:
                    filtered_candidates[skill_code] = data
                    logger.debug(f"  ✓ {skill_name} ({skill_code}) - タイプ: {skill_type} -> 採用")
                else:
                    rejected_count += 1
                    logger.debug(f"  ✗ {skill_name} ({skill_code}) - タイプ: {skill_type} -> 除外")

            candidates = filtered_candidates
            logger.info(f"フィルタ後の候補数: {len(candidates)} (除外: {rejected_count}件)")

        # スコアでソート
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:n]

        # Recommendationオブジェクトに変換
        recommendations = []
        for rank, (skill_code, data) in enumerate(sorted_candidates, 1):
            skill_name = self.get_skill_name(skill_code)
            explanation = self.explain(member_code, skill_code)

            # 信頼度の計算（パス品質を考慮）
            confidence = self._calculate_confidence(data)

            recommendations.append(Recommendation(
                skill_code=skill_code,
                skill_name=skill_name,
                score=data['score'],
                rank=rank,
                explanation=explanation,
                confidence=confidence,
                metadata=data
            ))

        return recommendations

    def _calculate_path_quality(self,
                                source: str,
                                target: str,
                                path_length: int) -> float:
        """
        パス品質スコアを計算（0-1）

        考慮要素:
        - パス長（短いほど良い）
        - 遷移強度（多いほど良い）
        - 遷移の安定性（標準偏差が小さいほど良い）
        - 新規性（最近の遷移があるほど良い）

        Returns:
            品質スコア（0-1）
        """
        if (source, target) not in self.edge_details:
            return 0.5  # デフォルト

        edge_info = self.edge_details[(source, target)]

        # パス長スコア（2が最適、長いほど減点）
        if path_length <= 2:
            length_score = 1.0
        else:
            length_score = max(0.3, 1.0 - (path_length - 2) * 0.15)

        # 遷移強度スコア（対数スケール）
        count = edge_info['count']
        strength_score = min(1.0, np.log1p(count) / np.log1p(50))

        # 安定性スコア（標準偏差が小さいほど良い）
        std_days = edge_info['std_days']
        if std_days <= 30:
            stability_score = 1.0
        else:
            stability_score = max(0.3, 1.0 - (std_days - 30) / 180)

        # 新規性スコア（最近の遷移があるほど良い）
        recency_days = edge_info['recency_days']
        if recency_days <= 90:
            recency_score = 1.0
        elif recency_days <= 365:
            recency_score = 0.8
        else:
            recency_score = max(0.3, 1.0 - (recency_days - 365) / 730)

        # 重み付き平均
        quality = (
            0.25 * length_score +
            0.30 * strength_score +
            0.25 * stability_score +
            0.20 * recency_score
        )

        return quality

    def _get_embedding_recommendations(self,
                                      user_skills: List[str],
                                      exclude_acquired: bool,
                                      n: int) -> Dict[str, float]:
        """
        埋め込み空間での推薦（Robust Scaling適用）

        Returns:
            {skill_code: score}
        """
        all_similarities = {}

        for skill in user_skills:
            if skill not in self.node2vec_model.wv:
                continue

            try:
                similar_skills = self.node2vec_model.wv.most_similar(skill, topn=n)
                for sim_skill, similarity in similar_skills:
                    if exclude_acquired and sim_skill in user_skills:
                        continue

                    if sim_skill not in all_similarities:
                        all_similarities[sim_skill] = []
                    all_similarities[sim_skill].append(similarity)
            except Exception as e:
                logger.warning(f"類似スキル取得エラー ({skill}): {e}")

        # 平均類似度を計算
        avg_similarities = {
            skill: np.mean(sims) for skill, sims in all_similarities.items()
        }

        # Robust Scaling（外れ値に強い）
        if self.use_robust_scaling and avg_similarities:
            scores = list(avg_similarities.values())
            median = np.median(scores)
            q75, q25 = np.percentile(scores, [75, 25])
            iqr = q75 - q25

            if iqr > 0:
                scaled = {
                    skill: ((score - median) / iqr) * 5 + 5  # 0-10のスケール
                    for skill, score in avg_similarities.items()
                }
                return {k: max(0, v) for k, v in scaled.items()}

        # デフォルト: そのまま10倍
        return {k: v * 10 for k, v in avg_similarities.items()}

    def _calculate_confidence(self, data: Dict) -> float:
        """信頼度を計算"""
        if data['reason_type'] == 'direct_transition':
            # 遷移人数とパス品質から計算
            raw_count = data['metadata'].get('raw_count', 0)
            path_quality = data.get('path_quality', 0.5)

            count_confidence = min(1.0, raw_count / 10)
            confidence = 0.6 * count_confidence + 0.4 * path_quality
        else:
            # 埋め込み類似度から計算
            confidence = 0.6
            if 'has_embedding_similarity' in data:
                confidence = 0.7

        return confidence

    def explain(self, member_code: str, skill_code: str) -> str:
        """推薦理由を説明"""
        self._check_fitted()

        user_skills = self.get_user_skills(member_code)
        skill_name = self.get_skill_name(skill_code)

        explanations = []

        # 1. 直接遷移の説明（詳細版）
        for user_skill in user_skills:
            if self.graph.has_edge(user_skill, skill_code):
                source_name = self.get_skill_name(user_skill)

                if (user_skill, skill_code) in self.edge_details:
                    details = self.edge_details[(user_skill, skill_code)]
                    count = details['count']
                    median_days = details['median_days']
                    recency_days = details['recency_days']

                    recency_text = ""
                    if recency_days <= 90:
                        recency_text = "（最近の遷移実績あり）"

                    explanations.append(
                        f"🎯 {source_name}を習得した{count}人が次に{skill_name}を学習 "
                        f"（平均{median_days:.0f}日後）{recency_text}"
                    )

        # 2. 学習パスの説明
        for user_skill in user_skills:
            try:
                if user_skill in self.graph and skill_code in self.graph:
                    path = nx.shortest_path(self.graph, user_skill, skill_code)
                    if 2 <= len(path) <= 4:
                        path_names = [self.get_skill_name(s) for s in path]
                        path_str = " → ".join(path_names)

                        # パス品質を計算
                        quality = self._calculate_path_quality(
                            user_skill, skill_code, len(path)
                        )
                        quality_text = ""
                        if quality >= 0.7:
                            quality_text = "（高品質パス）"

                        explanations.append(f"📚 推奨学習パス: {path_str} {quality_text}")
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        if not explanations:
            return f"グラフ構造から{skill_name}が推薦されました"

        return "\n".join(explanations)

    def get_edge_statistics(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """エッジの詳細統計を取得"""
        return self.edge_details.copy()

    def _get_skill_type(self, skill_code: str) -> str:
        """
        スキルの力量タイプを取得

        Args:
            skill_code: スキルコード

        Returns:
            力量タイプ（'SKILL', 'EDUCATION', 'LICENSE', 'UNKNOWN'）
        """
        if self.competence_master is None:
            return 'UNKNOWN'

        # 力量コードの検索（日本語・英語両対応）
        code_col = '力量コード' if '力量コード' in self.competence_master.columns else 'competence_code'
        skill_row = self.competence_master[
            self.competence_master[code_col] == skill_code
        ]

        if skill_row.empty:
            return 'UNKNOWN'

        # 力量タイプカラムの検索（日本語・英語両対応）
        if '力量タイプ' in skill_row.columns:
            return skill_row.iloc[0]['力量タイプ']
        elif 'competence_type' in skill_row.columns:
            return skill_row.iloc[0]['competence_type']
        else:
            return 'UNKNOWN'
