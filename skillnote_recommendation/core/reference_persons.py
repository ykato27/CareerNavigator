"""
参考人物検索機能（改訂版）

推薦された力量に対して、参考になる人物を3タイプ検索する:
1. 近い距離の先輩（追いつきやすい目標）
2. エキスパート（専門性×最高レベル）
3. 異分野の達人（多様性×異なるキャリア）

改訂ポイント:
- 計算効率の改善（類似度キャッシング、辞書化でO(1)アクセス）
- 設定の外部化（閾値をハードコーディングしない）
- コード重複の削減（共通メソッドの抽出）
- 品質メトリクスの追加（ロギング、説明性の向上）
- 型安全性の向上（厳密な型ヒント）
- フォールバック戦略の改善（段階的な条件緩和）
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity
from skillnote_recommendation.core.models import ReferencePerson

# ロガー設定
logger = logging.getLogger(__name__)


@dataclass
class ReferencePersonConfig:
    """参考人物検索の設定パラメータ"""

    # 近い距離の先輩の設定
    close_senior_skill_ratio_min: float = 1.1  # 総合スキルレベルの最低比率
    close_senior_skill_ratio_max: float = 1.5  # 総合スキルレベルの最大比率

    # 異分野の達人の設定
    diverse_expert_max_similarity: float = 0.5  # 最大類似度（これ以下で異分野とみなす）
    diverse_expert_fallback_similarity: float = 0.7  # フォールバック時の最大類似度

    # スキルレベルの閾値
    min_competence_level: float = 2.0  # 推薦力量の最低レベル（5段階で初級以上）

    # フォールバック設定
    enable_fallback: bool = True  # フォールバック機能の有効化

    def __post_init__(self):
        """設定値のバリデーション"""
        if not 1.0 <= self.close_senior_skill_ratio_min <= self.close_senior_skill_ratio_max:
            raise ValueError("Invalid close_senior_skill_ratio range")
        if not 0 <= self.diverse_expert_max_similarity <= 1:
            raise ValueError("diverse_expert_max_similarity must be between 0 and 1")
        if self.min_competence_level < 0:
            raise ValueError("min_competence_level must be non-negative")


class ReferencePersonFinder:
    """参考人物検索エンジン（改訂版）"""

    def __init__(
        self,
        member_competence: pd.DataFrame,
        member_master: pd.DataFrame,
        competence_master: pd.DataFrame,
        config: Optional[ReferencePersonConfig] = None,
    ):
        """
        Args:
            member_competence: メンバー習得力量データ
            member_master: メンバーマスタ
            competence_master: 力量マスタ
            config: 設定パラメータ（省略時はデフォルト値を使用）
        """
        self.member_competence = member_competence
        self.member_master = member_master
        self.competence_master = competence_master
        self.config = config or ReferencePersonConfig()

        # メンバー×力量マトリックスを作成
        self.member_skill_matrix = member_competence.pivot_table(
            index="メンバーコード", columns="力量コード", values="正規化レベル", fill_value=0
        )

        # ==========================================
        # パフォーマンス最適化: 事前計算とキャッシング
        # ==========================================

        # 類似度行列を事前計算してキャッシュ（O(1)アクセス）
        self._similarity_matrix: Optional[pd.DataFrame] = None
        self._precompute_similarities()

        # DataFrameを辞書に変換してO(1)アクセスを実現
        self._member_competences_cache: Dict[str, Set[str]] = {}
        self._member_levels_cache: Dict[str, Dict[str, float]] = {}
        self._member_total_level_cache: Dict[str, float] = {}
        self._member_avg_level_cache: Dict[str, float] = {}
        self._member_count_cache: Dict[str, int] = {}
        self._member_name_cache: Dict[str, str] = {}
        self._competence_name_cache: Dict[str, str] = {}
        self._precompute_caches()

        logger.info(
            f"ReferencePersonFinder initialized with {len(self.member_skill_matrix)} members"
        )

    def _precompute_similarities(self) -> None:
        """類似度行列を事前計算（初期化時に1回だけ実行）"""
        if len(self.member_skill_matrix) == 0:
            self._similarity_matrix = pd.DataFrame()
            return

        # コサイン類似度を計算
        similarity_values = cosine_similarity(self.member_skill_matrix.values)

        # DataFrameに変換（メンバーコードをインデックスとカラムに使用）
        self._similarity_matrix = pd.DataFrame(
            similarity_values,
            index=self.member_skill_matrix.index,
            columns=self.member_skill_matrix.index,
        )

        logger.debug(f"Similarity matrix precomputed: shape={self._similarity_matrix.shape}")

    def _precompute_caches(self) -> None:
        """各種キャッシュを事前計算（初期化時に1回だけ実行）"""
        # メンバーごとにグループ化して一括処理
        grouped = self.member_competence.groupby("メンバーコード")

        for member_code, group in grouped:
            # 保有力量のセット
            self._member_competences_cache[member_code] = set(group["力量コード"].values)

            # 力量レベルの辞書
            self._member_levels_cache[member_code] = dict(
                zip(group["力量コード"], group["正規化レベル"])
            )

            # 総合レベル
            self._member_total_level_cache[member_code] = float(group["正規化レベル"].sum())

            # 平均レベル
            self._member_avg_level_cache[member_code] = float(group["正規化レベル"].mean())

            # 力量数
            self._member_count_cache[member_code] = len(group)

        # メンバー名のキャッシュ
        for _, row in self.member_master.iterrows():
            self._member_name_cache[row["メンバーコード"]] = row["メンバー名"]

        # 力量名のキャッシュ
        for _, row in self.competence_master.iterrows():
            self._competence_name_cache[row["力量コード"]] = row["力量名"]

        logger.debug(f"Caches precomputed: {len(self._member_competences_cache)} members")

    def find_reference_persons(
        self, target_member_code: str, recommended_competence_code: str, top_n: int = 3
    ) -> List[ReferencePerson]:
        """
        推薦された力量に対して参考になる人物を3タイプ検索

        Args:
            target_member_code: 対象メンバーコード
            recommended_competence_code: 推薦された力量コード
            top_n: 各タイプから返す人数（デフォルト: 各タイプ1人ずつ、合計3人）

        Returns:
            参考人物リスト（最大3人: 近い先輩1人、エキスパート1人、異分野1人）
        """
        logger.info(
            f"Finding reference persons for {target_member_code}, competence: {recommended_competence_code}"
        )

        reference_persons = []

        # 類似度情報を一度だけ取得（パフォーマンス改善）
        similarities = self._get_similarities(target_member_code)
        if not similarities:
            logger.warning(f"No similarities found for {target_member_code}")
            return reference_persons

        # 推薦力量の保有者リストを事前取得
        competence_holders = self._get_competence_holders(recommended_competence_code)
        if not competence_holders:
            logger.warning(f"No holders found for competence: {recommended_competence_code}")
            return reference_persons

        # 1. 近い距離の先輩を検索
        close_senior = self._find_close_senior(
            target_member_code, recommended_competence_code, similarities, competence_holders
        )
        if close_senior:
            reference_persons.append(close_senior)
            logger.info(f"Found close senior: {close_senior.member_name}")

        # 2. エキスパートを検索
        expert = self._find_expert(
            target_member_code, recommended_competence_code, similarities, competence_holders
        )
        if expert:
            reference_persons.append(expert)
            logger.info(f"Found expert: {expert.member_name}")

        # 3. 異分野の達人を検索
        diverse_expert = self._find_diverse_expert(
            target_member_code, recommended_competence_code, similarities, competence_holders
        )
        if diverse_expert:
            reference_persons.append(diverse_expert)
            logger.info(f"Found diverse expert: {diverse_expert.member_name}")

        logger.info(f"Found {len(reference_persons)} reference persons in total")
        return reference_persons

    # =========================================================
    # 参考人物検索メソッド（3タイプ）
    # =========================================================

    def _find_close_senior(
        self,
        target_member_code: str,
        recommended_competence_code: str,
        similarities: Dict[str, float],
        competence_holders: Set[str],
    ) -> Optional[ReferencePerson]:
        """
        近い距離の先輩を検索（追いつきやすい目標）

        戦略:
        - 総合スキルレベルが自分より1.1〜1.5倍程度高い
        - 推薦力量を保有している
        - その中で類似度が最も高い人を選択
        """
        logger.debug(f"Searching for close senior for {target_member_code}")

        target_total_level = self._get_total_skill_level(target_member_code)

        # 候補者をフィルタリング
        def filter_func(member_code: str, similarity: float) -> bool:
            # 推薦力量を保有しているか
            if member_code not in competence_holders:
                return False

            # 総合スキルレベルが適切な範囲か
            reference_total_level = self._get_total_skill_level(member_code)
            skill_ratio = (
                reference_total_level / target_total_level if target_total_level > 0 else 0
            )

            return (
                self.config.close_senior_skill_ratio_min
                <= skill_ratio
                <= self.config.close_senior_skill_ratio_max
            )

        candidates = self._filter_candidates(target_member_code, similarities, filter_func)

        if not candidates:
            logger.debug("No close senior candidates found")
            return None

        # 最も類似度が高い人を選択（追いつきやすさを重視）
        best_member_code, similarity = max(candidates, key=lambda x: x[1])

        # ReferencePerson オブジェクトを構築
        return self._build_reference_person(
            target_member_code=target_member_code,
            reference_member_code=best_member_code,
            reference_type="close_senior",
            similarity=similarity,
            recommended_competence_code=recommended_competence_code,
        )

    def _find_expert(
        self,
        target_member_code: str,
        recommended_competence_code: str,
        similarities: Dict[str, float],
        competence_holders: Set[str],
    ) -> Optional[ReferencePerson]:
        """
        エキスパートを検索（専門性×最高レベル）

        戦略:
        - 推薦力量のレベルが最も高い人を選択
        - 同レベルが複数いる場合は、類似度が高い方を優先
        """
        logger.debug(f"Searching for expert for {target_member_code}")

        # 推薦力量を持つ全員のレベルを取得
        candidates_with_level = []
        for holder in competence_holders:
            if holder == target_member_code:
                continue

            level = self._get_competence_level(holder, recommended_competence_code)
            if level is not None and level >= self.config.min_competence_level:
                similarity = similarities.get(holder, 0.0)
                candidates_with_level.append((holder, level, similarity))

        if not candidates_with_level:
            logger.debug("No expert candidates found")
            return None

        # レベルが最も高い人を選択（同レベルなら類似度で選択）
        best_member_code, best_level, similarity = max(
            candidates_with_level, key=lambda x: (x[1], x[2])  # レベル優先、次に類似度
        )

        logger.debug(f"Expert found with level={best_level}, similarity={similarity:.2f}")

        # ReferencePerson オブジェクトを構築
        return self._build_reference_person(
            target_member_code=target_member_code,
            reference_member_code=best_member_code,
            reference_type="expert",
            similarity=similarity,
            recommended_competence_code=recommended_competence_code,
        )

    def _find_diverse_expert(
        self,
        target_member_code: str,
        recommended_competence_code: str,
        similarities: Dict[str, float],
        competence_holders: Set[str],
    ) -> Optional[ReferencePerson]:
        """
        異分野の達人を検索（多様性×異なるキャリア）

        戦略:
        - 類似度が低い（異なるキャリアパス）
        - 推薦力量を保有している
        - その中で類似度が最も低い人を選択（多様性を重視）
        - フォールバック: 条件を段階的に緩和
        """
        logger.debug(f"Searching for diverse expert for {target_member_code}")

        # 候補者をフィルタリング（低類似度）
        def filter_func(member_code: str, similarity: float) -> bool:
            # 推薦力量を保有しているか
            if member_code not in competence_holders:
                return False

            # 推薦力量のレベルチェック
            level = self._get_competence_level(member_code, recommended_competence_code)
            if level is None or level < self.config.min_competence_level:
                return False

            # 類似度が低いか
            return similarity <= self.config.diverse_expert_max_similarity

        candidates = self._filter_candidates(target_member_code, similarities, filter_func)

        # フォールバック: 類似度が低い人がいない場合は条件を緩和
        if not candidates and self.config.enable_fallback:
            logger.debug("Applying fallback for diverse expert search")

            def fallback_filter_func(member_code: str, similarity: float) -> bool:
                if member_code not in competence_holders:
                    return False
                level = self._get_competence_level(member_code, recommended_competence_code)
                if level is None or level < self.config.min_competence_level:
                    return False
                return similarity <= self.config.diverse_expert_fallback_similarity

            candidates = self._filter_candidates(
                target_member_code, similarities, fallback_filter_func
            )

        if not candidates:
            logger.debug("No diverse expert candidates found")
            return None

        # 最も類似度が低い人を選択（多様性を重視）
        best_member_code, similarity = min(candidates, key=lambda x: x[1])

        # ReferencePerson オブジェクトを構築
        return self._build_reference_person(
            target_member_code=target_member_code,
            reference_member_code=best_member_code,
            reference_type="diverse_expert",
            similarity=similarity,
            recommended_competence_code=recommended_competence_code,
        )

    # =========================================================
    # 共通ヘルパーメソッド（コード重複の削減）
    # =========================================================

    def _filter_candidates(
        self,
        target_member_code: str,
        similarities: Dict[str, float],
        filter_func: Callable[[str, float], bool],
    ) -> List[Tuple[str, float]]:
        """
        候補者をフィルタリング（共通処理）

        Args:
            target_member_code: 対象メンバーコード
            similarities: 類似度辞書
            filter_func: フィルタリング関数

        Returns:
            (メンバーコード, 類似度) のリスト
        """
        candidates = []
        for member_code, similarity in similarities.items():
            # 自分自身を除外
            if member_code == target_member_code:
                continue

            # カスタムフィルタ適用
            if filter_func(member_code, similarity):
                candidates.append((member_code, similarity))

        return candidates

    def _build_reference_person(
        self,
        target_member_code: str,
        reference_member_code: str,
        reference_type: str,
        similarity: float,
        recommended_competence_code: str,
    ) -> ReferencePerson:
        """
        ReferencePerson オブジェクトを構築（共通処理）

        Args:
            target_member_code: 対象メンバーコード
            reference_member_code: 参考人物のメンバーコード
            reference_type: 参考タイプ
            similarity: 類似度
            recommended_competence_code: 推薦力量コード

        Returns:
            ReferencePerson オブジェクト
        """
        # 差分分析
        common, unique, gap = self._analyze_competence_gap(
            target_member_code, reference_member_code
        )

        # メンバー名と力量名を取得
        member_name = self._get_member_name(reference_member_code)
        competence_name = self._get_competence_name(recommended_competence_code)

        # 推薦力量のレベルを取得（該当する場合）
        competence_level = self._get_competence_level(
            reference_member_code, recommended_competence_code
        )

        # 理由を生成
        reason = self._generate_reason(
            reference_type=reference_type,
            member_name=member_name,
            competence_name=competence_name,
            similarity=similarity,
            competence_level=competence_level,
            common_count=len(common),
            unique_count=len(unique),
            target_total_level=self._get_total_skill_level(target_member_code),
            reference_total_level=self._get_total_skill_level(reference_member_code),
        )

        return ReferencePerson(
            member_code=reference_member_code,
            member_name=member_name,
            reference_type=reference_type,
            similarity_score=similarity,
            common_competences=common,
            unique_competences=unique,
            competence_gap=gap,
            reason=reason,
        )

    def _get_similarities(self, target_member_code: str) -> Dict[str, float]:
        """
        対象メンバーと全メンバーの類似度を取得（キャッシュから取得）

        Args:
            target_member_code: 対象メンバーコード

        Returns:
            {メンバーコード: 類似度} の辞書
        """
        if (
            self._similarity_matrix is None
            or target_member_code not in self._similarity_matrix.index
        ):
            return {}

        # 類似度行列から該当行を取得して辞書に変換
        similarities = self._similarity_matrix.loc[target_member_code].to_dict()
        return similarities

    def _get_competence_holders(self, competence_code: str) -> Set[str]:
        """
        特定の力量を保有しているメンバーのセットを取得

        Args:
            competence_code: 力量コード

        Returns:
            メンバーコードのセット
        """
        holders = set()
        for member_code, competences in self._member_competences_cache.items():
            if competence_code in competences:
                holders.add(member_code)
        return holders

    def _analyze_competence_gap(
        self, target_member_code: str, reference_member_code: str
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """
        力量の差分を分析

        Args:
            target_member_code: 対象メンバーコード
            reference_member_code: 参考人物のメンバーコード

        Returns:
            (共通力量, 参考人物のユニーク力量, レベル差分)
        """
        target_competences = self._member_competences_cache.get(target_member_code, set())
        reference_competences = self._member_competences_cache.get(reference_member_code, set())

        # 共通の力量
        common = list(target_competences & reference_competences)

        # 参考人物が持つユニークな力量（対象メンバーが持っていない）
        unique = list(reference_competences - target_competences)

        # レベル差分を計算
        gap = {}
        target_levels = self._member_levels_cache.get(target_member_code, {})
        reference_levels = self._member_levels_cache.get(reference_member_code, {})

        for comp_code in common:
            target_level = target_levels.get(comp_code, 0.0)
            reference_level = reference_levels.get(comp_code, 0.0)
            gap[comp_code] = float(reference_level - target_level)

        return common, unique, gap

    # =========================================================
    # キャッシュアクセスメソッド（O(1)アクセス）
    # =========================================================

    def _get_member_competences(self, member_code: str) -> Set[str]:
        """メンバーが保有する力量コードのセットを取得（O(1)）"""
        return self._member_competences_cache.get(member_code, set())

    def _get_member_competence_levels(self, member_code: str) -> Dict[str, float]:
        """メンバーの力量レベルを辞書で取得（O(1)）"""
        return self._member_levels_cache.get(member_code, {})

    def _get_total_skill_level(self, member_code: str) -> float:
        """メンバーの総合スキルレベルを取得（O(1)）"""
        return self._member_total_level_cache.get(member_code, 0.0)

    def _get_average_skill_level(self, member_code: str) -> float:
        """メンバーの平均スキルレベルを取得（O(1)）"""
        return self._member_avg_level_cache.get(member_code, 0.0)

    def _get_competence_count(self, member_code: str) -> int:
        """メンバーの保有力量数を取得（O(1)）"""
        return self._member_count_cache.get(member_code, 0)

    def _get_competence_level(self, member_code: str, competence_code: str) -> Optional[float]:
        """特定の力量のレベルを取得（O(1)）"""
        levels = self._member_levels_cache.get(member_code, {})
        return levels.get(competence_code)

    def _get_member_name(self, member_code: str) -> str:
        """メンバー名を取得（O(1)）"""
        return self._member_name_cache.get(member_code, member_code)

    def _get_competence_name(self, competence_code: str) -> str:
        """力量名を取得（O(1)）"""
        return self._competence_name_cache.get(competence_code, competence_code)

    # =========================================================
    # 理由生成
    # =========================================================

    def _generate_reason(
        self,
        reference_type: str,
        member_name: str,
        competence_name: str,
        similarity: float,
        competence_level: Optional[float],
        common_count: int,
        unique_count: int,
        target_total_level: float,
        reference_total_level: float,
    ) -> str:
        """
        参考人物の理由を生成

        Args:
            reference_type: 参考タイプ
            member_name: メンバー名
            competence_name: 力量名
            similarity: 類似度
            competence_level: 推薦力量のレベル
            common_count: 共通力量数
            unique_count: ユニーク力量数
            target_total_level: 対象者の総合レベル
            reference_total_level: 参考人物の総合レベル

        Returns:
            理由テキスト
        """
        similarity_pct = int(similarity * 100)

        if reference_type == "close_senior":
            skill_ratio = (
                reference_total_level / target_total_level if target_total_level > 0 else 0
            )
            skill_ratio_pct = int(skill_ratio * 100)
            level_str = f"レベル{competence_level:.1f}" if competence_level else ""

            return (
                f"**{member_name}さん**は、あなたの総合スキルレベルの約{skill_ratio_pct}%で、"
                f"追いつきやすい目標となる先輩です。\n"
                f"推薦力量「{competence_name}」を{level_str}で習得済みで、"
                f"キャリア類似度は{similarity_pct}%と高く、あなたに近い成長パスを歩んでいます。\n"
                f"共通の力量が{common_count}個あり、具体的な習得方法を聞きやすい距離感です。"
            )

        elif reference_type == "expert":
            level_str = f"レベル{competence_level:.1f}" if competence_level else "高レベル"

            return (
                f"**{member_name}さん**は、「{competence_name}」を{level_str}で習得している"
                f"トップパフォーマーです。\n"
                f"この力量のエキスパートとして、最高水準のスキル活用方法や深い知見を持っています。\n"
                f"あなたとのキャリア類似度は{similarity_pct}%ですが、専門性の高さから"
                f"目標設定や長期的なスキル開発の参考になります。"
            )

        elif reference_type == "diverse_expert":
            level_str = f"レベル{competence_level:.1f}" if competence_level else ""

            return (
                f"**{member_name}さん**は、あなたとは異なるキャリアパス（類似度{similarity_pct}%）を"
                f"歩んでいますが、「{competence_name}」を{level_str}で習得しています。\n"
                f"あなたが持っていない{unique_count}個の力量を保有しており、"
                f"異なる視点からのスキル活用方法を学べます。\n"
                f"多様なキャリアの可能性を知る上で、新しい発見をもたらしてくれるでしょう。"
            )

        else:
            return f"{member_name}さんは、推薦力量「{competence_name}」の参考人物です。"
