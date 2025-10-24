"""
参考人物検索機能

推薦された力量に対して、参考になる人物を3タイプ検索する:
1. 類似キャリアの人物（似た力量セットを持つ人）
2. ロールモデル（推薦力量を持つ&似た傾向の人）
3. 異なるキャリアパスの人（異なる力量セットだが推薦力量を持つ人）
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from skillnote_recommendation.core.models import ReferencePerson


class ReferencePersonFinder:
    """参考人物検索エンジン"""

    def __init__(self, member_competence: pd.DataFrame, member_master: pd.DataFrame,
                 competence_master: pd.DataFrame):
        """
        Args:
            member_competence: 会員習得力量データ
            member_master: 会員マスタ
            competence_master: 力量マスタ
        """
        self.member_competence = member_competence
        self.member_master = member_master
        self.competence_master = competence_master

        # 会員×力量マトリクスを作成
        self.member_skill_matrix = member_competence.pivot_table(
            index="メンバーコード",
            columns="力量コード",
            values="正規化レベル",
            fill_value=0
        )

        # 類似度行列を事前計算
        self._similarity_matrix = None

    def find_reference_persons(self, target_member_code: str,
                               recommended_competence_code: str,
                               top_n: int = 3) -> List[ReferencePerson]:
        """
        推薦された力量に対して参考になる人物を3タイプ検索

        Args:
            target_member_code: 対象会員コード
            recommended_competence_code: 推薦された力量コード
            top_n: 各タイプから返す人数（デフォルト: 各タイプ1人ずつ、合計3人）

        Returns:
            参考人物リスト（最大3人: 類似1人、ロールモデル1人、異キャリア1人）
        """
        reference_persons = []

        # 1. 類似キャリアの人物を検索
        similar_person = self._find_similar_career_person(
            target_member_code, recommended_competence_code
        )
        if similar_person:
            reference_persons.append(similar_person)

        # 2. ロールモデルを検索
        role_model = self._find_role_model(
            target_member_code, recommended_competence_code
        )
        if role_model:
            reference_persons.append(role_model)

        # 3. 異なるキャリアパスの人を検索
        diverse_person = self._find_diverse_career_person(
            target_member_code, recommended_competence_code
        )
        if diverse_person:
            reference_persons.append(diverse_person)

        return reference_persons

    def _find_similar_career_person(self, target_member_code: str,
                                     recommended_competence_code: str) -> Optional[ReferencePerson]:
        """類似キャリアの人物を検索（推薦力量を持っている必要はない）"""
        # 類似度を計算
        similarities = self._calculate_similarities(target_member_code)

        # 自分自身と推薦力量を既に持っている人を除外
        target_has_competences = set(self._get_member_competences(target_member_code))

        candidates = []
        for member_code, similarity in similarities:
            if member_code == target_member_code:
                continue
            # 上位者（自分よりスキルレベルが高い人）のみを候補とする
            if not self._is_higher_skilled(member_code, target_member_code):
                continue
            # 類似度が高い人を優先
            if similarity > 0.3:  # 最低限の類似度
                candidates.append((member_code, similarity))

        if not candidates:
            return None

        # 最も類似度が高い人を選択
        best_member_code, similarity = max(candidates, key=lambda x: x[1])

        # 差分分析
        common, unique, gap = self._analyze_competence_gap(
            target_member_code, best_member_code
        )

        # 理由を生成
        member_name = self._get_member_name(best_member_code)
        reason = self._generate_similar_career_reason(
            member_name, similarity, len(common), len(unique)
        )

        return ReferencePerson(
            member_code=best_member_code,
            member_name=member_name,
            reference_type="similar_career",
            similarity_score=similarity,
            common_competences=common,
            unique_competences=unique,
            competence_gap=gap,
            reason=reason
        )

    def _find_role_model(self, target_member_code: str,
                         recommended_competence_code: str) -> Optional[ReferencePerson]:
        """ロールモデルを検索（推薦力量を持つ&類似の人）"""
        # 推薦力量を持っている人を検索
        holders = self.member_competence[
            self.member_competence["力量コード"] == recommended_competence_code
        ]["メンバーコード"].unique().tolist()

        if not holders:
            return None

        # その中で類似度が高い人を検索
        similarities = self._calculate_similarities(target_member_code)
        similarity_dict = dict(similarities)

        candidates = []
        for holder in holders:
            if holder == target_member_code:
                continue
            # 上位者（自分よりスキルレベルが高い人）のみを候補とする
            if not self._is_higher_skilled(holder, target_member_code):
                continue
            if holder in similarity_dict:
                candidates.append((holder, similarity_dict[holder]))

        if not candidates:
            return None

        # 最も類似度が高いロールモデルを選択
        best_member_code, similarity = max(candidates, key=lambda x: x[1])

        # 差分分析
        common, unique, gap = self._analyze_competence_gap(
            target_member_code, best_member_code
        )

        # 推薦力量のレベルを取得
        role_model_level = self.member_competence[
            (self.member_competence["メンバーコード"] == best_member_code) &
            (self.member_competence["力量コード"] == recommended_competence_code)
        ]["正規化レベル"].values[0]

        # 理由を生成
        member_name = self._get_member_name(best_member_code)
        competence_name = self._get_competence_name(recommended_competence_code)
        reason = self._generate_role_model_reason(
            member_name, competence_name, role_model_level, similarity
        )

        return ReferencePerson(
            member_code=best_member_code,
            member_name=member_name,
            reference_type="role_model",
            similarity_score=similarity,
            common_competences=common,
            unique_competences=unique,
            competence_gap=gap,
            reason=reason
        )

    def _find_diverse_career_person(self, target_member_code: str,
                                     recommended_competence_code: str) -> Optional[ReferencePerson]:
        """異なるキャリアパスの人を検索（低類似度だが推薦力量を持つ人）"""
        # 推薦力量を持っている人を検索
        holders = self.member_competence[
            self.member_competence["力量コード"] == recommended_competence_code
        ]["メンバーコード"].unique().tolist()

        if not holders:
            return None

        # その中で類似度が低い人を検索（異なるキャリアパス）
        similarities = self._calculate_similarities(target_member_code)
        similarity_dict = dict(similarities)

        candidates = []
        for holder in holders:
            if holder == target_member_code:
                continue
            # 上位者（自分よりスキルレベルが高い人）のみを候補とする
            if not self._is_higher_skilled(holder, target_member_code):
                continue
            if holder in similarity_dict:
                sim = similarity_dict[holder]
                # 類似度が低い人を優先（異なるキャリア）
                if sim < 0.5:  # 類似度が低い
                    candidates.append((holder, sim))

        if not candidates:
            # 類似度が低い人がいない場合は、中程度の人を選択
            candidates = []
            for holder in holders:
                if holder == target_member_code:
                    continue
                # 上位者（自分よりスキルレベルが高い人）のみを候補とする
                if not self._is_higher_skilled(holder, target_member_code):
                    continue
                if holder in similarity_dict:
                    candidates.append((holder, similarity_dict[holder]))

        if not candidates:
            return None

        # 最も類似度が低い人を選択（多様性を重視）
        best_member_code, similarity = min(candidates, key=lambda x: x[1])

        # 差分分析
        common, unique, gap = self._analyze_competence_gap(
            target_member_code, best_member_code
        )

        # 理由を生成
        member_name = self._get_member_name(best_member_code)
        competence_name = self._get_competence_name(recommended_competence_code)
        reason = self._generate_diverse_career_reason(
            member_name, competence_name, len(unique), similarity
        )

        return ReferencePerson(
            member_code=best_member_code,
            member_name=member_name,
            reference_type="diverse_career",
            similarity_score=similarity,
            common_competences=common,
            unique_competences=unique,
            competence_gap=gap,
            reason=reason
        )

    # =========================================================
    # ヘルパー関数
    # =========================================================

    def _calculate_similarities(self, target_member_code: str) -> List[Tuple[str, float]]:
        """対象会員と全会員の類似度を計算"""
        if target_member_code not in self.member_skill_matrix.index:
            return []

        # 対象会員のベクトル
        target_vector = self.member_skill_matrix.loc[target_member_code].values.reshape(1, -1)

        # 全会員との類似度を計算
        similarities = cosine_similarity(target_vector, self.member_skill_matrix.values)[0]

        # (会員コード, 類似度) のリストを作成
        result = []
        for idx, member_code in enumerate(self.member_skill_matrix.index):
            result.append((member_code, similarities[idx]))

        # 類似度の降順でソート
        result.sort(key=lambda x: x[1], reverse=True)

        return result

    def _analyze_competence_gap(self, target_member_code: str,
                                 reference_member_code: str) -> Tuple[List[str], List[str], Dict[str, int]]:
        """
        力量の差分を分析

        Returns:
            (共通力量, 参考人物のユニーク力量, レベル差分)
        """
        target_competences = set(self._get_member_competences(target_member_code))
        reference_competences = set(self._get_member_competences(reference_member_code))

        # 共通の力量
        common = list(target_competences & reference_competences)

        # 参考人物が持つユニークな力量（対象会員が持っていない）
        unique = list(reference_competences - target_competences)

        # レベル差分を計算
        gap = {}
        target_levels = self._get_member_competence_levels(target_member_code)
        reference_levels = self._get_member_competence_levels(reference_member_code)

        for comp_code in common:
            target_level = target_levels.get(comp_code, 0)
            reference_level = reference_levels.get(comp_code, 0)
            gap[comp_code] = reference_level - target_level

        return common, unique, gap

    def _get_member_competences(self, member_code: str) -> List[str]:
        """会員が保有する力量コードのリストを取得"""
        return self.member_competence[
            self.member_competence["メンバーコード"] == member_code
        ]["力量コード"].unique().tolist()

    def _get_member_competence_levels(self, member_code: str) -> Dict[str, int]:
        """会員の力量レベルを辞書で取得"""
        df = self.member_competence[
            self.member_competence["メンバーコード"] == member_code
        ]
        return dict(zip(df["力量コード"], df["正規化レベル"]))

    def _get_total_skill_level(self, member_code: str) -> float:
        """会員の総合スキルレベルを取得（正規化レベルの合計）"""
        df = self.member_competence[
            self.member_competence["メンバーコード"] == member_code
        ]
        return df["正規化レベル"].sum()

    def _is_higher_skilled(self, reference_member_code: str, target_member_code: str) -> bool:
        """参考人物が対象者より上位者（スキルレベルが高い）かどうかを判定"""
        target_level = self._get_total_skill_level(target_member_code)
        reference_level = self._get_total_skill_level(reference_member_code)
        return reference_level > target_level

    def _get_member_name(self, member_code: str) -> str:
        """会員名を取得"""
        member = self.member_master[
            self.member_master["メンバーコード"] == member_code
        ]
        if len(member) > 0:
            return member.iloc[0]["メンバー名"]
        return member_code

    def _get_competence_name(self, competence_code: str) -> str:
        """力量名を取得"""
        comp = self.competence_master[
            self.competence_master["力量コード"] == competence_code
        ]
        if len(comp) > 0:
            return comp.iloc[0]["力量名"]
        return competence_code

    # =========================================================
    # 理由生成
    # =========================================================

    def _generate_similar_career_reason(self, member_name: str, similarity: float,
                                         common_count: int, unique_count: int) -> str:
        """類似キャリアの人の理由を生成"""
        similarity_pct = int(similarity * 100)
        reason = (
            f"**{member_name}さん**は、あなたと{similarity_pct}%類似したキャリアパスを歩んでいます。\n"
            f"共通の力量が{common_count}個あり、あなたが今後習得を目指す力量を含めて"
            f"{unique_count}個の追加力量を保有しています。\n"
            f"同じような傾向を持つ先輩として、キャリア形成の参考になるでしょう。"
        )
        return reason

    def _generate_role_model_reason(self, member_name: str, competence_name: str,
                                     level: int, similarity: float) -> str:
        """ロールモデルの理由を生成"""
        similarity_pct = int(similarity * 100)
        reason = (
            f"**{member_name}さん**は、推薦力量「{competence_name}」を"
            f"レベル{level}で既に習得しています。\n"
            f"あなたとのキャリア類似度は{similarity_pct}%で、似た傾向を持ちながら"
            f"この力量を活用している良いロールモデルです。\n"
            f"どのように習得したか、どう活用しているかを参考にしてみてください。"
        )
        return reason

    def _generate_diverse_career_reason(self, member_name: str, competence_name: str,
                                         unique_count: int, similarity: float) -> str:
        """異キャリアパスの人の理由を生成"""
        similarity_pct = int(similarity * 100)
        reason = (
            f"**{member_name}さん**は、あなたとは異なるキャリアパス（類似度{similarity_pct}%）を"
            f"歩んでいますが、「{competence_name}」を習得しています。\n"
            f"あなたが持っていない{unique_count}個の力量を保有しており、"
            f"将来的に異なる方向性を目指す場合の参考になります。\n"
            f"多様なキャリアの可能性を知る上で、刺激的なモデルケースとなるでしょう。"
        )
        return reason
