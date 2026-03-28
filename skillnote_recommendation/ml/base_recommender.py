"""
推薦モデルの基底クラス

全ての推薦モデルはこのBaseRecommenderを継承し、
統一されたインターフェースを提供します。
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """推薦結果の単位"""

    skill_code: str
    skill_name: str
    score: float
    rank: int
    explanation: str
    confidence: float
    metadata: Dict[str, Any]


class BaseRecommender(ABC):
    """
    推薦モデルの基底クラス

    全ての推薦モデルはこのクラスを継承し、以下のメソッドを実装する必要があります：
    - fit(): モデルの学習
    - recommend(): 推薦リストの生成
    - explain(): 推薦理由の説明
    """

    def __init__(self, name: str, interpretability_score: int = 3):
        """
        初期化

        Args:
            name: モデル名
            interpretability_score: 解釈性スコア（1-5、5が最も解釈しやすい）
        """
        self.name = name
        self.interpretability_score = interpretability_score
        self.is_fitted = False
        self.metadata = {}

    @abstractmethod
    def fit(self, member_competence: pd.DataFrame, competence_master: pd.DataFrame) -> None:
        """
        モデルの学習

        Args:
            member_competence: メンバー習得力量データ
            competence_master: 力量マスタデータ
        """
        pass

    @abstractmethod
    def recommend(
        self, member_code: str, n: int = 10, exclude_acquired: bool = True
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
        pass

    @abstractmethod
    def explain(self, member_code: str, skill_code: str) -> str:
        """
        推薦理由の説明

        Args:
            member_code: メンバーコード
            skill_code: スキルコード

        Returns:
            推薦理由の説明文
        """
        pass

    def get_user_skills(self, member_code: str) -> List[str]:
        """
        ユーザーが既に習得しているスキルを取得

        Args:
            member_code: メンバーコード

        Returns:
            習得済みスキルコードのリスト
        """
        if not hasattr(self, "member_competence"):
            return []

        user_data = self.member_competence[self.member_competence["メンバーコード"] == member_code]

        # レベル > 0 のスキルを習得済みとみなす
        if "正規化レベル" in user_data.columns:
            acquired = user_data[user_data["正規化レベル"] > 0]
        elif "レベル" in user_data.columns:
            acquired = user_data[user_data["レベル"] > 0]
        else:
            acquired = user_data

        return acquired["力量コード"].tolist()

    def get_skill_name(self, skill_code: str) -> str:
        """
        スキルコードからスキル名を取得

        Args:
            skill_code: スキルコード

        Returns:
            スキル名（見つからない場合はスキルコード）
        """
        if not hasattr(self, "competence_master"):
            return skill_code

        skill = self.competence_master[self.competence_master["力量コード"] == skill_code]

        if len(skill) > 0 and "力量名" in skill.columns:
            return skill.iloc[0]["力量名"]

        return skill_code

    def get_interpretability_info(self) -> Dict[str, Any]:
        """
        解釈性に関する情報を取得

        Returns:
            解釈性スコアと説明
        """
        interpretability_levels = {
            5: "非常に高い - 推薦理由が直感的に理解できる",
            4: "高い - 推薦理由を明確に説明できる",
            3: "中程度 - 一部の推薦理由を説明できる",
            2: "低い - 推薦理由の説明が難しい",
            1: "非常に低い - ブラックボックス",
        }

        return {
            "score": self.interpretability_score,
            "level": interpretability_levels.get(self.interpretability_score, "不明"),
            "model_name": self.name,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデルに関する情報を取得

        Returns:
            モデル情報の辞書
        """
        return {
            "name": self.name,
            "is_fitted": self.is_fitted,
            "interpretability_score": self.interpretability_score,
            "metadata": self.metadata,
        }

    def _check_fitted(self) -> None:
        """モデルが学習済みかチェック"""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name}は学習されていません。まずfit()メソッドを呼び出してください。"
            )
