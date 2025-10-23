"""
推薦実行メインスクリプト

変換済みデータを使用して力量推薦を実行
"""

import sys
import os

# srcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from recommendation_system import RecommendationSystem


def main():
    """メイン処理"""
    print("=" * 80)
    print("スキルノート 推薦システム")
    print("=" * 80)
    
    try:
        # 推薦システム初期化
        system = RecommendationSystem()
        
        # サンプル実行: 中程度の習得数の会員を選択
        member_stats = system.df_member_competence.groupby('メンバーコード').size().sort_values()
        sample_code = member_stats.index[len(member_stats)//2]
        
        print("\n" + "=" * 80)
        print("サンプル実行1: 全タイプの力量を推薦")
        print("=" * 80)
        system.print_recommendations(sample_code, top_n=10)
        
        print("\n" + "=" * 80)
        print("サンプル実行2: SKILLタイプのみを推薦")
        print("=" * 80)
        system.print_recommendations(sample_code, competence_type='SKILL', top_n=5)
        
        print("\n" + "=" * 80)
        print("処理完了")
        print("=" * 80)
        print("\nカスタム実行方法:")
        print("  from src.recommendation_system import RecommendationSystem")
        print("  system = RecommendationSystem()")
        print("  system.print_recommendations('会員コード', top_n=10)")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nエラー: 変換済みデータが見つかりません")
        print(f"先に convert_data.py を実行してください")
        print(f"\n  python convert_data.py")
        return 1
    
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
