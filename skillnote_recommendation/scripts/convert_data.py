"""
データ変換スクリプト

CSVファイルを読み込み、推薦システム用にデータを変換
"""

import sys
from skillnote_recommendation.core.config import Config
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.core.similarity_calculator import SimilarityCalculator


def main():
    """メイン処理"""
    print("=" * 80)
    print("スキルノート データ変換処理")
    print("=" * 80)
    print(f"\n入力ディレクトリ: {Config.DATA_DIR}")
    print(f"出力ディレクトリ: {Config.OUTPUT_DIR}")
    
    # ディレクトリ作成
    Config.ensure_directories()
    
    try:
        # ステップ1: データ読み込み
        loader = DataLoader()
        data = loader.load_all_data()
        
        # データ検証
        if not loader.validate_data(data):
            print("\nエラー: データ検証に失敗しました")
            return 1
        
        # ステップ2: データ変換
        transformer = DataTransformer()
        
        # 統合力量マスタ作成
        competence_master = transformer.create_competence_master(data)
        
        # 会員習得力量データ作成
        member_competence, valid_members = transformer.create_member_competence(
            data, competence_master
        )
        
        # 会員×力量マトリクス作成
        skill_matrix = transformer.create_skill_matrix(member_competence)
        
        # 会員マスタクリーニング
        members_clean = transformer.clean_members_data(data)
        
        # ステップ3: 類似度計算
        similarity_calc = SimilarityCalculator()
        similarity_df = similarity_calc.calculate_similarity(member_competence)
        
        # ステップ4: データ保存
        print("\n" + "=" * 80)
        print("データ保存")
        print("=" * 80)
        
        members_clean.to_csv(
            Config.get_output_path('members_clean'),
            index=False,
            encoding=Config.OUTPUT_ENCODING
        )
        print(f"  ✓ {Config.OUTPUT_FILES['members_clean']}: {len(members_clean)}件")
        
        competence_master.to_csv(
            Config.get_output_path('competence_master'),
            index=False,
            encoding=Config.OUTPUT_ENCODING
        )
        print(f"  ✓ {Config.OUTPUT_FILES['competence_master']}: {len(competence_master)}件")
        
        member_competence.to_csv(
            Config.get_output_path('member_competence'),
            index=False,
            encoding=Config.OUTPUT_ENCODING
        )
        print(f"  ✓ {Config.OUTPUT_FILES['member_competence']}: {len(member_competence)}件")
        
        skill_matrix.to_csv(
            Config.get_output_path('skill_matrix'),
            encoding=Config.OUTPUT_ENCODING
        )
        print(f"  ✓ {Config.OUTPUT_FILES['skill_matrix']}: "
              f"{skill_matrix.shape[0]}×{skill_matrix.shape[1]}")
        
        similarity_df.to_csv(
            Config.get_output_path('competence_similarity'),
            index=False,
            encoding=Config.OUTPUT_ENCODING
        )
        print(f"  ✓ {Config.OUTPUT_FILES['competence_similarity']}: {len(similarity_df)}件")
        
        print("\n" + "=" * 80)
        print("データ変換処理完了")
        print("=" * 80)
        print(f"\n変換データは {Config.OUTPUT_DIR}/ に保存されました")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nエラー: {e}")
        print(f"\n必要なCSVファイルを {Config.DATA_DIR}/ に配置してください:")
        for key, filename in Config.INPUT_FILES.items():
            print(f"  - {filename}")
        return 1
    
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
