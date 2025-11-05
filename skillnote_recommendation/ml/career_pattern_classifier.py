"""
キャリアパターン分類器

メンバーを類似度に基づいて3つのキャリアパターンに分類：
1. 類似キャリア：類似度が高い（0.7以上）
2. 異なるキャリア1：類似度が中程度（0.4〜0.7）
3. 異なるキャリア2：類似度が低い（0.4未満）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CareerPatternGroup:
    """キャリアパターングループ"""
    pattern_name: str  # 'similar', 'different1', 'different2'
    pattern_label: str  # 表示用ラベル
    member_codes: List[str]  # メンバーコードのリスト
    member_names: List[str]  # メンバー名のリスト
    similarities: List[float]  # 類似度のリスト


class CareerPatternClassifier:
    """キャリアパターン分類器"""

    def __init__(
        self,
        member_competence: pd.DataFrame,
        member_master: pd.DataFrame,
        mf_model,
        similar_threshold: float = 0.7,
        different1_threshold: float = 0.4,
        max_persons_per_group: int = 5,
        min_persons_per_group: int = 3
    ):
        """
        初期化

        Args:
            member_competence: メンバー習得力量データ
            member_master: メンバーマスタ
            mf_model: Matrix Factorizationモデル
            similar_threshold: 類似キャリアの閾値
            different1_threshold: 異なるキャリア1の閾値
            max_persons_per_group: 各グループの最大人数
            min_persons_per_group: 各グループの最小人数
        """
        self.member_competence = member_competence
        self.member_master = member_master
        self.mf_model = mf_model
        self.similar_threshold = similar_threshold
        self.different1_threshold = different1_threshold
        self.max_persons_per_group = max_persons_per_group
        self.min_persons_per_group = min_persons_per_group

    def classify_career_patterns(
        self,
        target_member_code: str
    ) -> Dict[str, CareerPatternGroup]:
        """
        対象メンバーに対して、他のメンバーを3つのキャリアパターンに分類

        Args:
            target_member_code: 対象メンバーコード

        Returns:
            {'similar': group1, 'different1': group2, 'different2': group3}
        """
        # 対象メンバーの潜在因子ベクトルを取得
        if target_member_code not in self.mf_model.member_index:
            raise ValueError(f"メンバーコード '{target_member_code}' が見つかりません。")

        target_factors = self.mf_model.get_member_factors(target_member_code)
        target_factors = target_factors.reshape(1, -1)

        # すべてのメンバーの類似度を計算
        similarities = []
        for member_code in self.mf_model.member_codes:
            if member_code == target_member_code:
                continue  # 自分自身は除外

            member_factors = self.mf_model.get_member_factors(member_code)
            member_factors = member_factors.reshape(1, -1)

            # コサイン類似度を計算
            similarity = cosine_similarity(target_factors, member_factors)[0][0]
            similarities.append((member_code, similarity))

        # 類似度でソート（降順）
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 3つのグループに分類
        similar_group = []
        different1_group = []
        different2_group = []

        for member_code, similarity in similarities:
            if similarity >= self.similar_threshold:
                similar_group.append((member_code, similarity))
            elif similarity >= self.different1_threshold:
                different1_group.append((member_code, similarity))
            else:
                different2_group.append((member_code, similarity))

        # 各グループから上位を選択
        similar_group = similar_group[:self.max_persons_per_group]
        different1_group = different1_group[:self.max_persons_per_group]
        different2_group = different2_group[:self.max_persons_per_group]

        # CareerPatternGroupオブジェクトに変換
        result = {
            'similar': self._create_group(
                'similar',
                '💼 類似キャリア',
                similar_group
            ),
            'different1': self._create_group(
                'different1',
                '🌟 異なるキャリア1',
                different1_group
            ),
            'different2': self._create_group(
                'different2',
                '🚀 異なるキャリア2',
                different2_group
            )
        }

        return result

    def _create_group(
        self,
        pattern_name: str,
        pattern_label: str,
        member_list: List[Tuple[str, float]]
    ) -> CareerPatternGroup:
        """
        CareerPatternGroupオブジェクトを作成

        Args:
            pattern_name: パターン名
            pattern_label: 表示用ラベル
            member_list: (メンバーコード, 類似度) のリスト

        Returns:
            CareerPatternGroup
        """
        member_codes = []
        member_names = []
        similarities = []

        for member_code, similarity in member_list:
            member_codes.append(member_code)
            similarities.append(similarity)

            # メンバー名を取得
            member_info = self.member_master[
                self.member_master["メンバーコード"] == member_code
            ]
            if len(member_info) > 0:
                member_names.append(member_info.iloc[0]["メンバー名"])
            else:
                member_names.append(member_code)

        return CareerPatternGroup(
            pattern_name=pattern_name,
            pattern_label=pattern_label,
            member_codes=member_codes,
            member_names=member_names,
            similarities=similarities
        )

    def get_group_statistics(
        self,
        groups: Dict[str, CareerPatternGroup]
    ) -> Dict[str, dict]:
        """
        各グループの統計情報を取得

        Args:
            groups: classify_career_patterns() の結果

        Returns:
            統計情報の辞書
        """
        stats = {}

        for pattern_name, group in groups.items():
            stats[pattern_name] = {
                'count': len(group.member_codes),
                'avg_similarity': np.mean(group.similarities) if group.similarities else 0,
                'min_similarity': np.min(group.similarities) if group.similarities else 0,
                'max_similarity': np.max(group.similarities) if group.similarities else 0,
            }

        return stats


def create_classifier_from_config(
    member_competence: pd.DataFrame,
    member_master: pd.DataFrame,
    mf_model,
    config
) -> CareerPatternClassifier:
    """
    Configからキャリアパターン分類器を作成

    Args:
        member_competence: メンバー習得力量データ
        member_master: メンバーマスタ
        mf_model: Matrix Factorizationモデル
        config: Configクラスインスタンス

    Returns:
        CareerPatternClassifier
    """
    params = config.CAREER_PATTERN_PARAMS

    return CareerPatternClassifier(
        member_competence=member_competence,
        member_master=member_master,
        mf_model=mf_model,
        similar_threshold=params['similar_career_threshold'],
        different1_threshold=params['different_career1_threshold'],
        max_persons_per_group=params['similar_career_ref_persons'],  # 全グループで同じ値を使用
        min_persons_per_group=params['min_ref_persons']
    )
