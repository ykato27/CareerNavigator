"""
推薦システム

外部から使いやすいファサードインターフェースを提供
"""

import os
import pandas as pd
from typing import List, Optional
from skillnote_recommendation.core.models import Recommendation
from skillnote_recommendation.core.recommendation_engine import RecommendationEngine
from skillnote_recommendation.core.config import Config


class RecommendationSystem:
    """推薦システムクラス（ファサード）"""
    
    def __init__(
        self,
        output_dir: str = None,
        df_members: pd.DataFrame = None,
        df_competence_master: pd.DataFrame = None,
        df_member_competence: pd.DataFrame = None,
        df_similarity: pd.DataFrame = None
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ（Noneの場合はConfig.OUTPUT_DIRを使用）
            df_members: 会員マスタのDataFrame
            df_competence_master: 力量マスタのDataFrame
            df_member_competence: 会員習得力量のDataFrame
            df_similarity: 力量類似度マトリクス（任意）
        """
        self.output_dir = output_dir or Config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("推薦システム初期化")
        print("=" * 80)

        # DataFrameが直接渡された場合はファイル読み込みをスキップ
        if df_members is not None:
            self.df_members = df_members
            self.df_competence_master = df_competence_master
            self.df_member_competence = df_member_competence
            self.df_similarity = df_similarity if df_similarity is not None else pd.DataFrame()
        else:
            # 従来の動作（Config経由でファイル読み込み）
            self.df_members = pd.read_csv(Config.get_output_path('members_clean'), encoding=Config.FILE_ENCODING)
            self.df_competence_master = pd.read_csv(Config.get_output_path('competence_master'), encoding=Config.FILE_ENCODING)
            self.df_member_competence = pd.read_csv(Config.get_output_path('member_competence'), encoding=Config.FILE_ENCODING)
            self.df_similarity = pd.read_csv(Config.get_output_path('competence_similarity'), encoding=Config.FILE_ENCODING)

        # 情報ログ
        print(f"\n  会員数: {len(self.df_members)}")
        print(f"  力量数: {len(self.df_competence_master)}")
        print(f"  習得記録数: {len(self.df_member_competence)}")
        print("  初期化完了")

        # 推薦エンジン初期化
        self.engine = RecommendationEngine(
            self.df_members,
            self.df_competence_master,
            self.df_member_competence,
            self.df_similarity
        )

    # ---- 以降は既存のまま ---- #

    def get_member_info(self, member_code: str) -> dict:
        member = self.df_members[self.df_members['メンバーコード'] == member_code]
        if len(member) == 0:
            return None
        member = member.iloc[0]
        acquired = self.engine.get_member_competences(member_code)
        skill_count = len(acquired[acquired['力量タイプ'] == 'SKILL'])
        edu_count = len(acquired[acquired['力量タイプ'] == 'EDUCATION'])
        lic_count = len(acquired[acquired['力量タイプ'] == 'LICENSE'])
        return {
            'member_code': member_code,
            'name': member['メンバー名'],
            'role': member['役職'] if pd.notna(member['役職']) else '未設定',
            'grade': member['職能・等級'] if pd.notna(member['職能・等級']) else '未設定',
            'skill_count': skill_count,
            'education_count': edu_count,
            'license_count': lic_count
        }

    def recommend_competences(self, member_code: str, competence_type: Optional[str] = None,
                              category_filter: Optional[str] = None, top_n: int = 10) -> List[Recommendation]:
        return self.engine.recommend(member_code, competence_type, category_filter, top_n)

    def export_recommendations(self, member_code: str, output_file: str,
                               competence_type: Optional[str] = None, top_n: int = 20):
        recommendations = self.recommend_competences(member_code, competence_type, top_n=top_n)
        if not recommendations:
            print("推薦できる力量がありません")
            return
        data = [rec.to_dict() for rec in recommendations]
        df = pd.DataFrame(data)
        output_path = os.path.join(self.output_dir, output_file)
        df.to_csv(output_path, index=False, encoding=Config.OUTPUT_ENCODING)
        print(f"推薦結果を {output_path} に出力しました")
