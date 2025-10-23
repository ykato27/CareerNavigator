"""
推薦システム

外部から使いやすいファサードインターフェースを提供
"""

import pandas as pd
from typing import List, Optional
from models import Recommendation
from recommendation_engine import RecommendationEngine
from config import Config


class RecommendationSystem:
    """推薦システムクラス（ファサード）"""
    
    def __init__(self, output_dir: str = None):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ（Noneの場合はConfig.OUTPUT_DIRを使用）
        """
        self.output_dir = output_dir or Config.OUTPUT_DIR
        
        print("\n" + "=" * 80)
        print("推薦システム初期化")
        print("=" * 80)
        
        # データ読み込み
        self.df_members = pd.read_csv(
            Config.get_output_path('members_clean'),
            encoding=Config.FILE_ENCODING
        )
        self.df_competence_master = pd.read_csv(
            Config.get_output_path('competence_master'),
            encoding=Config.FILE_ENCODING
        )
        self.df_member_competence = pd.read_csv(
            Config.get_output_path('member_competence'),
            encoding=Config.FILE_ENCODING
        )
        self.df_similarity = pd.read_csv(
            Config.get_output_path('competence_similarity'),
            encoding=Config.FILE_ENCODING
        )
        
        print(f"\n  会員数: {len(self.df_members)}")
        print(f"  力量数: {len(self.df_competence_master)}")
        print(f"  習得記録数: {len(self.df_member_competence)}")
        print("  初期化完了")
        
        # 推薦エンジンを初期化
        self.engine = RecommendationEngine(
            self.df_members,
            self.df_competence_master,
            self.df_member_competence,
            self.df_similarity
        )
    
    def get_member_info(self, member_code: str) -> dict:
        """
        会員情報を取得
        
        Args:
            member_code: 会員コード
            
        Returns:
            会員情報の辞書
        """
        member = self.df_members[self.df_members['メンバーコード'] == member_code]
        
        if len(member) == 0:
            return None
        
        member = member.iloc[0]
        
        # 保有力量数を集計
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
    
    def recommend_competences(self, member_code: str, 
                             competence_type: Optional[str] = None,
                             category_filter: Optional[str] = None,
                             top_n: int = 10) -> List[Recommendation]:
        """
        力量を推薦
        
        Args:
            member_code: 会員コード
            competence_type: 力量タイプフィルタ（None/SKILL/EDUCATION/LICENSE）
            category_filter: カテゴリフィルタ（部分一致）
            top_n: 推薦件数
            
        Returns:
            推薦結果のリスト
        """
        return self.engine.recommend(member_code, competence_type, category_filter, top_n)
    
    def print_recommendations(self, member_code: str, 
                            competence_type: Optional[str] = None,
                            top_n: int = 10):
        """
        推薦結果を整形して出力
        
        Args:
            member_code: 会員コード
            competence_type: 力量タイプフィルタ
            top_n: 推薦件数
        """
        # 会員情報取得
        member_info = self.get_member_info(member_code)
        
        if member_info is None:
            print(f"会員コード {member_code} が見つかりません")
            return
        
        print("\n" + "=" * 80)
        print("力量推薦結果")
        print("=" * 80)
        print(f"会員: {member_info['name']} ({member_info['member_code']})")
        print(f"役職: {member_info['role']}")
        print(f"職能等級: {member_info['grade']}")
        print(f"\n保有力量: SKILL {member_info['skill_count']}件 / "
              f"EDUCATION {member_info['education_count']}件 / "
              f"LICENSE {member_info['license_count']}件")
        print("=" * 80)
        
        # 推薦取得
        type_label = f"（{competence_type}のみ）" if competence_type else "（全タイプ）"
        print(f"\n推薦力量 {type_label}（上位{top_n}件）:\n")
        
        recommendations = self.recommend_competences(member_code, competence_type, top_n=top_n)
        
        if not recommendations:
            print("推薦できる力量がありません")
            return
        
        for i, rec in enumerate(recommendations, 1):
            print(f"【推薦 {i}】 {rec.competence_name}")
            print(f"  タイプ: {rec.competence_type}")
            print(f"  カテゴリ: {rec.category}")
            print(f"  優先度スコア: {rec.priority_score:.2f}")
            print(f"  推薦理由: {rec.reason}")
            print()
        
        print("=" * 80)
    
    def export_recommendations(self, member_code: str, output_file: str,
                              competence_type: Optional[str] = None,
                              top_n: int = 20):
        """
        推薦結果をCSVファイルに出力
        
        Args:
            member_code: 会員コード
            output_file: 出力ファイル名
            competence_type: 力量タイプフィルタ
            top_n: 推薦件数
        """
        recommendations = self.recommend_competences(member_code, competence_type, top_n=top_n)
        
        if not recommendations:
            print("推薦できる力量がありません")
            return
        
        # 辞書形式に変換
        data = [rec.to_dict() for rec in recommendations]
        df = pd.DataFrame(data)
        
        # 出力
        import os
        output_path = os.path.join(self.output_dir, output_file)
        df.to_csv(output_path, index=False, encoding=Config.OUTPUT_ENCODING)
        
        print(f"推薦結果を {output_path} に出力しました")
