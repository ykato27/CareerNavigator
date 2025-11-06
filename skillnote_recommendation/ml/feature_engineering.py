"""
特徴量エンジニアリング

職種・等級のワンホットエンコーディング、習得力量の時系列パターン抽出、
力量間の共起関係、カテゴリ階層の埋め込み表現などを実装
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict, Counter
from datetime import datetime


class FeatureEngineer:
    """特徴量エンジニアリングクラス

    職種・等級などのメンバー属性と力量データから、
    推薦に有用な特徴量を抽出する
    """

    def __init__(self,
                 member_master: pd.DataFrame,
                 competence_master: pd.DataFrame,
                 member_competence: pd.DataFrame,
                 category_hierarchy: Optional[Dict] = None):
        """
        Args:
            member_master: メンバーマスタ (member_code, name, role, grade)
            competence_master: 力量マスタ (competence_code, name, competence_type, category)
            member_competence: メンバー習得力量 (member_code, competence_code, level, acquired_date)
            category_hierarchy: カテゴリ階層構造 {parent: [children]}
        """
        self.member_master = member_master
        self.competence_master = competence_master
        self.member_competence = member_competence
        self.category_hierarchy = category_hierarchy or {}

        # エンコーダーの初期化
        self.role_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.grade_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.category_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # フィッティング
        self._fit_encoders()

        # 力量共起関係の計算
        self.competence_cooccurrence = self._compute_competence_cooccurrence()

        # カテゴリ埋め込みの計算
        self.category_embeddings = self._compute_category_embeddings()

        print("\n特徴量エンジニアリング初期化完了")
        print(f"  職種数: {len(self.role_encoder.categories_[0])}")
        print(f"  等級数: {len(self.grade_encoder.categories_[0])}")
        print(f"  カテゴリ数: {len(self.category_encoder.categories_[0])}")
        print(f"  力量タイプ数: {len(self.type_encoder.categories_[0])}")

    def _fit_encoders(self):
        """エンコーダーをフィッティング"""
        # 職種と等級のエンコーダー
        if 'role' in self.member_master.columns:
            roles = self.member_master['role'].fillna('UNKNOWN').values.reshape(-1, 1)
            self.role_encoder.fit(roles)

        if 'grade' in self.member_master.columns:
            grades = self.member_master['grade'].fillna('UNKNOWN').values.reshape(-1, 1)
            self.grade_encoder.fit(grades)

        # カテゴリと力量タイプのエンコーダー
        if 'category' in self.competence_master.columns:
            categories = self.competence_master['category'].fillna('UNKNOWN').values.reshape(-1, 1)
            self.category_encoder.fit(categories)

        if 'competence_type' in self.competence_master.columns:
            types = self.competence_master['competence_type'].fillna('SKILL').values.reshape(-1, 1)
            self.type_encoder.fit(types)

    def encode_member_attributes(self, member_code: str) -> np.ndarray:
        """メンバー属性をワンホットエンコーディング

        Args:
            member_code: メンバーコード

        Returns:
            ワンホットエンコードされた特徴ベクトル
        """
        member_info = self.member_master[
            self.member_master['member_code'] == member_code
        ]

        if member_info.empty:
            # デフォルト値を返す
            role_vec = self.role_encoder.transform([['UNKNOWN']])[0]
            grade_vec = self.grade_encoder.transform([['UNKNOWN']])[0]
        else:
            member_row = member_info.iloc[0]
            role = member_row.get('role', 'UNKNOWN')
            grade = member_row.get('grade', 'UNKNOWN')

            role_vec = self.role_encoder.transform([[role]])[0]
            grade_vec = self.grade_encoder.transform([[grade]])[0]

        # 結合して返す
        return np.concatenate([role_vec, grade_vec])

    def encode_competence_attributes(self, competence_code: str) -> np.ndarray:
        """力量属性をワンホットエンコーディング

        Args:
            competence_code: 力量コード

        Returns:
            ワンホットエンコードされた特徴ベクトル
        """
        comp_info = self.competence_master[
            self.competence_master['competence_code'] == competence_code
        ]

        if comp_info.empty:
            category_vec = self.category_encoder.transform([['UNKNOWN']])[0]
            type_vec = self.type_encoder.transform([['SKILL']])[0]
        else:
            comp_row = comp_info.iloc[0]
            category = comp_row.get('category', 'UNKNOWN')
            comp_type = comp_row.get('competence_type', 'SKILL')

            category_vec = self.category_encoder.transform([[category]])[0]
            type_vec = self.type_encoder.transform([[comp_type]])[0]

        return np.concatenate([category_vec, type_vec])

    def extract_temporal_patterns(self, member_code: str) -> Dict[str, float]:
        """習得力量の時系列パターンを抽出

        Args:
            member_code: メンバーコード

        Returns:
            時系列特徴量の辞書
            {
                'acquisition_rate': 力量習得率（過去6ヶ月）,
                'recent_activity': 直近活動度,
                'skill_variety': スキルの多様性,
                'category_focus': カテゴリの集中度,
                'learning_velocity': 学習速度
            }
        """
        member_comps = self.member_competence[
            self.member_competence['member_code'] == member_code
        ].copy()

        if member_comps.empty:
            return {
                'acquisition_rate': 0.0,
                'recent_activity': 0.0,
                'skill_variety': 0.0,
                'category_focus': 0.0,
                'learning_velocity': 0.0
            }

        # acquired_dateがある場合のみ時系列分析を実施
        if 'acquired_date' in member_comps.columns:
            member_comps['acquired_date'] = pd.to_datetime(
                member_comps['acquired_date'],
                errors='coerce'
            )
            member_comps = member_comps.dropna(subset=['acquired_date'])

            if not member_comps.empty:
                # 最新日付を基準に
                max_date = member_comps['acquired_date'].max()

                # 過去6ヶ月の習得率
                six_months_ago = max_date - pd.Timedelta(days=180)
                recent_comps = member_comps[member_comps['acquired_date'] >= six_months_ago]
                acquisition_rate = len(recent_comps) / 180.0  # 日あたりの習得率

                # 直近活動度（過去30日）
                thirty_days_ago = max_date - pd.Timedelta(days=30)
                very_recent = member_comps[member_comps['acquired_date'] >= thirty_days_ago]
                recent_activity = len(very_recent) / 30.0

                # 学習速度（日数あたりの習得数の変化率）
                if len(member_comps) > 1:
                    sorted_comps = member_comps.sort_values('acquired_date')
                    date_diffs = sorted_comps['acquired_date'].diff().dt.days
                    learning_velocity = 1.0 / (date_diffs.mean() + 1)  # 平均日数の逆数
                else:
                    learning_velocity = 0.0
            else:
                acquisition_rate = 0.0
                recent_activity = 0.0
                learning_velocity = 0.0
        else:
            acquisition_rate = 0.0
            recent_activity = 0.0
            learning_velocity = 0.0

        # スキルの多様性（カテゴリ数/全力量数）
        member_comps_with_category = member_comps.merge(
            self.competence_master[['competence_code', 'category']],
            on='competence_code',
            how='left'
        )
        unique_categories = member_comps_with_category['category'].nunique()
        total_competences = len(member_comps)
        skill_variety = unique_categories / max(total_competences, 1)

        # カテゴリの集中度（最頻カテゴリの割合）
        if not member_comps_with_category.empty:
            category_counts = member_comps_with_category['category'].value_counts()
            if len(category_counts) > 0:
                category_focus = category_counts.iloc[0] / total_competences
            else:
                category_focus = 0.0
        else:
            category_focus = 0.0

        return {
            'acquisition_rate': float(acquisition_rate),
            'recent_activity': float(recent_activity),
            'skill_variety': float(skill_variety),
            'category_focus': float(category_focus),
            'learning_velocity': float(learning_velocity)
        }

    def _compute_competence_cooccurrence(self) -> Dict[str, Dict[str, float]]:
        """力量間の共起関係を計算

        Returns:
            {competence_code: {other_competence_code: cooccurrence_score}}
        """
        cooccurrence = defaultdict(lambda: defaultdict(int))

        # メンバーごとに力量を集計
        member_groups = self.member_competence.groupby('member_code')['competence_code'].apply(list)

        # 共起をカウント
        for competences in member_groups:
            for i, comp1 in enumerate(competences):
                for comp2 in competences[i+1:]:
                    cooccurrence[comp1][comp2] += 1
                    cooccurrence[comp2][comp1] += 1

        # 正規化（Jaccard係数風）
        normalized_cooccurrence = {}
        for comp1, related in cooccurrence.items():
            normalized_cooccurrence[comp1] = {}
            comp1_count = len(self.member_competence[
                self.member_competence['competence_code'] == comp1
            ])

            for comp2, count in related.items():
                comp2_count = len(self.member_competence[
                    self.member_competence['competence_code'] == comp2
                ])
                # Jaccard係数: intersection / union
                union = comp1_count + comp2_count - count
                if union > 0:
                    normalized_cooccurrence[comp1][comp2] = count / union

        return normalized_cooccurrence

    def get_competence_cooccurrence_score(self,
                                         comp1: str,
                                         comp2: str) -> float:
        """2つの力量の共起スコアを取得

        Args:
            comp1: 力量コード1
            comp2: 力量コード2

        Returns:
            共起スコア（0-1）
        """
        return self.competence_cooccurrence.get(comp1, {}).get(comp2, 0.0)

    def get_related_competences(self,
                               competence_code: str,
                               top_k: int = 10) -> List[Tuple[str, float]]:
        """関連する力量を共起スコア順で取得

        Args:
            competence_code: 力量コード
            top_k: 上位K件

        Returns:
            [(related_competence_code, score), ...]
        """
        related = self.competence_cooccurrence.get(competence_code, {})
        sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
        return sorted_related[:top_k]

    def _compute_category_embeddings(self, embedding_dim: int = 16) -> Dict[str, np.ndarray]:
        """カテゴリ階層の埋め込み表現を計算

        シンプルなアプローチ: カテゴリに属する力量の統計情報を使用

        Args:
            embedding_dim: 埋め込み次元数

        Returns:
            {category: embedding_vector}
        """
        embeddings = {}

        # カテゴリごとに力量を集計
        category_groups = self.competence_master.groupby('category')

        for category, group in category_groups:
            comp_codes = group['competence_code'].tolist()

            # このカテゴリの力量を習得しているメンバー数
            member_counts = []
            for comp_code in comp_codes:
                count = len(self.member_competence[
                    self.member_competence['competence_code'] == comp_code
                ])
                member_counts.append(count)

            # 統計情報から埋め込みを生成
            if member_counts:
                features = [
                    np.mean(member_counts),       # 平均習得者数
                    np.std(member_counts),        # 標準偏差
                    np.median(member_counts),     # 中央値
                    np.max(member_counts),        # 最大値
                    np.min(member_counts),        # 最小値
                    len(comp_codes),              # カテゴリ内力量数
                ]

                # embedding_dimに合わせてパディングまたはトランケート
                if len(features) < embedding_dim:
                    features.extend([0.0] * (embedding_dim - len(features)))
                else:
                    features = features[:embedding_dim]

                embeddings[category] = np.array(features, dtype=np.float32)
            else:
                embeddings[category] = np.zeros(embedding_dim, dtype=np.float32)

        return embeddings

    def get_category_embedding(self, category: str) -> np.ndarray:
        """カテゴリの埋め込みベクトルを取得

        Args:
            category: カテゴリ名

        Returns:
            埋め込みベクトル
        """
        return self.category_embeddings.get(
            category,
            np.zeros(16, dtype=np.float32)
        )

    def compute_member_competence_affinity(self,
                                          member_code: str,
                                          competence_code: str) -> float:
        """メンバーと力量の親和性スコアを計算（コンテンツベース）

        メンバーの属性（職種・等級）と力量の属性（カテゴリ・タイプ）、
        既習得力量との共起関係などから、親和性を算出

        Args:
            member_code: メンバーコード
            competence_code: 力量コード

        Returns:
            親和性スコア（0-1）
        """
        # メンバーのエンコーディング
        member_vec = self.encode_member_attributes(member_code)

        # 力量のエンコーディング
        comp_vec = self.encode_competence_attributes(competence_code)

        # メンバーの既習得力量
        acquired_comps = self.member_competence[
            self.member_competence['member_code'] == member_code
        ]['competence_code'].tolist()

        # 共起スコアの平均
        cooccurrence_scores = []
        for acquired_comp in acquired_comps:
            score = self.get_competence_cooccurrence_score(acquired_comp, competence_code)
            cooccurrence_scores.append(score)

        avg_cooccurrence = np.mean(cooccurrence_scores) if cooccurrence_scores else 0.0

        # 時系列パターン
        temporal_features = self.extract_temporal_patterns(member_code)

        # 力量のカテゴリ情報
        comp_info = self.competence_master[
            self.competence_master['competence_code'] == competence_code
        ]

        if not comp_info.empty:
            comp_category = comp_info.iloc[0].get('category', 'UNKNOWN')

            # メンバーが習得済みの力量のカテゴリ分布
            acquired_categories = self.member_competence[
                self.member_competence['member_code'] == member_code
            ].merge(
                self.competence_master[['competence_code', 'category']],
                on='competence_code',
                how='left'
            )['category'].tolist()

            # このカテゴリの習得比率
            category_familiarity = acquired_categories.count(comp_category) / max(len(acquired_categories), 1)
        else:
            category_familiarity = 0.0

        # 親和性スコアの計算（重み付き平均）
        affinity_score = (
            0.3 * avg_cooccurrence +              # 共起関係
            0.2 * category_familiarity +           # カテゴリ親和性
            0.2 * temporal_features['skill_variety'] +  # スキル多様性
            0.15 * temporal_features['learning_velocity'] +  # 学習速度
            0.15 * temporal_features['recent_activity']      # 直近活動度
        )

        return min(max(affinity_score, 0.0), 1.0)  # 0-1にクリップ

    def compute_batch_affinity(self,
                              member_code: str,
                              competence_codes: List[str]) -> Dict[str, float]:
        """複数の力量に対して親和性スコアをバッチ計算

        Args:
            member_code: メンバーコード
            competence_codes: 力量コードのリスト

        Returns:
            {competence_code: affinity_score}
        """
        return {
            comp_code: self.compute_member_competence_affinity(member_code, comp_code)
            for comp_code in competence_codes
        }
