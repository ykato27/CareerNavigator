"""
グラフベース推薦モデル

スキル間の学習遷移をグラフで表現し、
Node2Vecで埋め込みを学習して推薦を行います。
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from datetime import datetime, timedelta

from skillnote_recommendation.ml.base_recommender import BaseRecommender, Recommendation

logger = logging.getLogger(__name__)


class SkillTransitionGraphRecommender(BaseRecommender):
    """
    スキル遷移グラフベースの推薦モデル

    メンバーの学習履歴からスキル間の遷移パターンを抽出し、
    グラフ構造として表現します。Node2Vecで埋め込みを学習し、
    学習パスに基づいた推薦を行います。
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
        workers: int = 4
    ):
        """
        初期化

        Args:
            time_window_days: 遷移とみなす最大期間（日数）
            min_transition_count: 最小遷移人数（この人数未満は除外）
            embedding_dim: 埋め込み次元数
            walk_length: ランダムウォークの長さ
            num_walks: ウォーク回数
            p: Return parameter（DFS vs BFS）
            q: In-out parameter（local vs global）
            workers: 並列処理数
        """
        super().__init__(
            name="SkillTransitionGraph",
            interpretability_score=4  # 高解釈性
        )

        self.time_window_days = time_window_days
        self.min_transition_count = min_transition_count
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers

        self.graph = None
        self.node2vec_model = None
        self.transition_stats = {}

    def fit(self, member_competence: pd.DataFrame, competence_master: pd.DataFrame) -> None:
        """
        モデルの学習

        Args:
            member_competence: メンバー習得力量データ
            competence_master: 力量マスタデータ
        """
        logger.info("=" * 80)
        logger.info("SkillTransitionGraphRecommender の学習を開始")
        logger.info("=" * 80)

        self.member_competence = member_competence.copy()
        self.competence_master = competence_master.copy()

        # 取得日カラムの確認
        if '取得日' not in member_competence.columns:
            raise ValueError("スキル遷移グラフには「取得日」カラムが必要です")

        # グラフ構築
        logger.info("\nStep 1: スキル遷移グラフの構築")
        self.graph = self._build_transition_graph()
        logger.info(f"  ノード数: {self.graph.number_of_nodes()}")
        logger.info(f"  エッジ数: {self.graph.number_of_edges()}")

        # Node2Vec学習
        if self.graph.number_of_nodes() > 1:
            logger.info("\nStep 2: Node2Vec埋め込みの学習")
            self._train_node2vec()
            logger.info(f"  埋め込み次元: {self.embedding_dim}")
            logger.info(f"  ウォーク長: {self.walk_length}")
            logger.info(f"  ウォーク回数: {self.num_walks}")
        else:
            logger.warning("グラフのノード数が不足しています。Node2Vecをスキップします。")

        self.is_fitted = True
        self.metadata = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'time_window_days': self.time_window_days,
            'min_transition_count': self.min_transition_count,
            'has_embeddings': self.node2vec_model is not None
        }

        logger.info("\n" + "=" * 80)
        logger.info("学習完了")
        logger.info("=" * 80)

    def _build_transition_graph(self) -> nx.DiGraph:
        """
        スキル遷移グラフを構築

        Returns:
            有向グラフ
        """
        G = nx.DiGraph()

        # 取得日を日付型に変換
        df = self.member_competence.copy()
        df['取得日_dt'] = pd.to_datetime(df['取得日'], errors='coerce')
        df = df[df['取得日_dt'].notna()]

        # メンバーごとに学習順序を抽出
        transition_counts = {}

        for member in df['メンバーコード'].unique():
            member_skills = df[
                df['メンバーコード'] == member
            ].sort_values('取得日_dt')

            skills = member_skills[['力量コード', '取得日_dt']].values

            # 連続するスキルペアを抽出
            for i in range(len(skills)):
                for j in range(i + 1, len(skills)):
                    source_skill, source_date = skills[i]
                    target_skill, target_date = skills[j]

                    # 時間窓内の遷移のみ
                    time_diff = (target_date - source_date).days
                    if time_diff <= self.time_window_days:
                        edge = (source_skill, target_skill)
                        if edge not in transition_counts:
                            transition_counts[edge] = {
                                'count': 0,
                                'time_diffs': []
                            }
                        transition_counts[edge]['count'] += 1
                        transition_counts[edge]['time_diffs'].append(time_diff)

        # 最小遷移回数以上のエッジのみグラフに追加
        for (source, target), stats in transition_counts.items():
            if stats['count'] >= self.min_transition_count:
                avg_time_diff = np.mean(stats['time_diffs'])
                median_time_diff = np.median(stats['time_diffs'])

                G.add_edge(
                    source,
                    target,
                    weight=stats['count'],
                    avg_time_diff=avg_time_diff,
                    median_time_diff=median_time_diff
                )

                # 統計情報を保存
                self.transition_stats[(source, target)] = {
                    'count': stats['count'],
                    'avg_days': avg_time_diff,
                    'median_days': median_time_diff
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

            # モデル学習
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
        exclude_acquired: bool = True
    ) -> List[Recommendation]:
        """
        推薦リストの生成

        Args:
            member_code: メンバーコード
            n: 推薦する件数
            exclude_acquired: 既習得スキルを除外するか

        Returns:
            推薦結果のリスト
        """
        self._check_fitted()

        # ユーザーが既に習得しているスキル
        user_skills = self.get_user_skills(member_code)

        if not user_skills:
            logger.warning(f"メンバー {member_code} は習得スキルがありません")
            return []

        # 候補スキルを収集
        candidates = {}

        # 方法1: グラフの隣接ノード（直接遷移）
        for skill in user_skills:
            if skill in self.graph:
                for neighbor in self.graph.neighbors(skill):
                    if exclude_acquired and neighbor in user_skills:
                        continue

                    edge_data = self.graph[skill][neighbor]
                    score = edge_data['weight']  # 遷移人数

                    if neighbor not in candidates or candidates[neighbor]['score'] < score:
                        candidates[neighbor] = {
                            'score': score,
                            'source_skill': skill,
                            'reason_type': 'direct_transition',
                            'metadata': edge_data
                        }

        # 方法2: 埋め込み空間での類似度
        if self.node2vec_model is not None:
            for skill in user_skills:
                if skill not in self.node2vec_model.wv:
                    continue

                try:
                    similar_skills = self.node2vec_model.wv.most_similar(skill, topn=n*2)
                    for sim_skill, similarity in similar_skills:
                        if exclude_acquired and sim_skill in user_skills:
                            continue

                        # 既に直接遷移で見つかっている場合は、スコアを加算
                        if sim_skill in candidates:
                            candidates[sim_skill]['score'] += similarity * 10
                            candidates[sim_skill]['similarity'] = similarity
                        else:
                            candidates[sim_skill] = {
                                'score': similarity * 10,
                                'source_skill': skill,
                                'reason_type': 'embedding_similarity',
                                'similarity': similarity,
                                'metadata': {}
                            }
                except Exception as e:
                    logger.warning(f"類似スキル取得エラー ({skill}): {e}")

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

            # 信頼度の計算
            if data['reason_type'] == 'direct_transition':
                # 遷移人数ベース
                confidence = min(data['score'] / 10, 1.0)
            else:
                # 類似度ベース
                confidence = data.get('similarity', 0.5)

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

    def explain(self, member_code: str, skill_code: str) -> str:
        """
        推薦理由を説明

        Args:
            member_code: メンバーコード
            skill_code: スキルコード

        Returns:
            推薦理由の説明文
        """
        self._check_fitted()

        user_skills = self.get_user_skills(member_code)
        skill_name = self.get_skill_name(skill_code)

        explanations = []

        # 1. 直接遷移の説明
        for user_skill in user_skills:
            if self.graph.has_edge(user_skill, skill_code):
                source_name = self.get_skill_name(user_skill)

                stats = self.transition_stats.get((user_skill, skill_code), {})
                count = stats.get('count', 0)
                median_days = stats.get('median_days', 0)

                explanations.append(
                    f"🎯 {source_name}を習得した人の多くが次に{skill_name}を学習しています "
                    f"（{count}人、平均{median_days:.0f}日後）"
                )

        # 2. 学習パスの説明
        for user_skill in user_skills:
            try:
                if user_skill in self.graph and skill_code in self.graph:
                    path = nx.shortest_path(self.graph, user_skill, skill_code)
                    if 2 <= len(path) <= 4:  # パスが適度な長さの場合のみ
                        path_names = [self.get_skill_name(s) for s in path]
                        path_str = " → ".join(path_names)
                        explanations.append(f"📚 推奨学習パス: {path_str}")
                        break  # 最初のパスのみ表示
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        # 3. 埋め込み類似度の説明
        if self.node2vec_model is not None:
            for user_skill in user_skills:
                try:
                    if user_skill in self.node2vec_model.wv and skill_code in self.node2vec_model.wv:
                        similarity = self.node2vec_model.wv.similarity(user_skill, skill_code)
                        if similarity > 0.5:
                            source_name = self.get_skill_name(user_skill)
                            explanations.append(
                                f"🔗 {source_name}と学習パターンが類似しています（類似度: {similarity:.0%}）"
                            )
                            break  # 最初の類似のみ表示
                except KeyError:
                    continue

        if not explanations:
            return f"グラフ構造から{skill_name}が推薦されました"

        return "\n".join(explanations)

    def get_learning_path(
        self,
        source_skill: str,
        target_skill: str,
        max_length: int = 5
    ) -> Optional[List[str]]:
        """
        2つのスキル間の学習パスを取得

        Args:
            source_skill: 始点スキル
            target_skill: 終点スキル
            max_length: 最大パス長

        Returns:
            スキルコードのリスト（パスが存在しない場合はNone）
        """
        self._check_fitted()

        try:
            path = nx.shortest_path(self.graph, source_skill, target_skill)
            if len(path) <= max_length:
                return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        return None

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        グラフの統計情報を取得

        Returns:
            統計情報の辞書
        """
        self._check_fitted()

        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
        }

        # 次数の統計
        in_degrees = [d for n, d in self.graph.in_degree()]
        out_degrees = [d for n, d in self.graph.out_degree()]

        stats['avg_in_degree'] = np.mean(in_degrees) if in_degrees else 0
        stats['avg_out_degree'] = np.mean(out_degrees) if out_degrees else 0

        # トップスキル
        if in_degrees:
            top_target_skills = sorted(
                self.graph.in_degree(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            stats['top_target_skills'] = [
                (self.get_skill_name(s), degree) for s, degree in top_target_skills
            ]

        if out_degrees:
            top_source_skills = sorted(
                self.graph.out_degree(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            stats['top_source_skills'] = [
                (self.get_skill_name(s), degree) for s, degree in top_source_skills
            ]

        return stats
