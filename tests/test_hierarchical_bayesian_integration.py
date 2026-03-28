"""
階層的ベイジアン推薦システムの統合テスト

実データを使用して全パイプラインをテストします。
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("階層的ベイジアン推薦システム - 統合テスト")
    print("=" * 80)
    
    # データファイルのパス
    data_dir = project_root / 'data'
    category_csv = data_dir / 'categories' / 'competence_category_skillnote.csv'
    skill_csv = data_dir / 'skills' / 'skill_skillnote.csv'
    acquired_csv = data_dir / 'acquired' / 'acquiredCompetenceLevel.csv'
    member_csv = data_dir / 'members' / 'member_skillnote.csv'
    
    print(f"\nデータファイル:")
    print(f"  カテゴリ: {category_csv}")
    print(f"  スキル: {skill_csv}")
    print(f"  取得済み力量: {acquired_csv}")
    print(f"  メンバー: {member_csv}")
    
    # データを読み込み
    print("\n" + "=" * 80)
    print("データ読み込み")
    print("=" * 80)
    
    try:
        member_competence = pd.read_csv(acquired_csv)
        competence_master = pd.read_csv(skill_csv)
        
        print(f"✅ 取得済み力量データ: {len(member_competence)}行")
        print(f"✅ 力量マスタ: {len(competence_master)}行")
        
        # ユーザー数とスキル数を確認
        n_users = member_competence['メンバーコード'].nunique()
        skill_data = member_competence[
            member_competence['力量タイプ  ###[competence_type]###'] == 'SKILL'
        ]
        n_skills = skill_data['力量コード'].nunique()
        
        print(f"  ユーザー数: {n_users}")
        print(f"  スキル数: {n_skills}")
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return
    
    # 推薦システムを初期化
    print("\n" + "=" * 80)
    print("推薦システムの初期化と学習")
    print("=" * 80)
    
    try:
        from skillnote_recommendation.ml.hierarchical_bayesian_recommender import (
            HierarchicalBayesianRecommender
        )
        
        recommender = HierarchicalBayesianRecommender(
            member_competence=member_competence,
            competence_master=competence_master,
            category_csv_path=str(category_csv),
            skill_csv_path=str(skill_csv),
            max_indegree=3,
            n_components=10
        )
        
        print("✅ 推薦システムを初期化しました")
        
        # 学習を実行
        print("\n学習を開始...")
        recommender.fit()
        
        print("✅ 学習が完了しました")
        
    except Exception as e:
        print(f"❌ 初期化・学習エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # サンプルユーザーで推薦を生成
    print("\n" + "=" * 80)
    print("推薦生成テスト")
    print("=" * 80)
    
    # 最初の5人のユーザーで推薦を生成
    sample_users = member_competence['メンバーコード'].unique()[:5]
    
    for i, user_code in enumerate(sample_users, 1):
        print(f"\n【ユーザー {i}: {user_code}】")
        
        try:
            # 推薦を生成
            recommendations = recommender.recommend(
                member_code=user_code,
                top_n=5
            )
            
            if recommendations:
                print(f"✅ {len(recommendations)}件の推薦を生成")
                print("\n推薦スキル:")
                for j, rec in enumerate(recommendations, 1):
                    print(f"  {j}. {rec['力量名']}")
                    print(f"     スコア: {rec['スコア']:.4f}")
                    print(f"     説明: {rec['説明']}")
                    print(f"     カテゴリ: {rec['カテゴリ']}")
            else:
                print("⚠️  推薦が生成されませんでした")
                
        except Exception as e:
            print(f"❌ 推薦生成エラー: {e}")
            import traceback
            traceback.print_exc()
    
    # 統計情報を表示
    print("\n" + "=" * 80)
    print("システム統計情報")
    print("=" * 80)
    
    if recommender.hierarchy:
        print(f"\nカテゴリ階層:")
        print(f"  L1カテゴリ数: {len(recommender.hierarchy.level1_categories)}")
        print(f"  L2カテゴリ数: {len(recommender.hierarchy.level2_categories)}")
        print(f"  L3カテゴリ数: {len(recommender.hierarchy.level3_categories)}")
        print(f"  総スキル数: {len(recommender.hierarchy.skill_to_category)}")
    
    if recommender.network_learner and recommender.network_learner.model:
        network_info = recommender.network_learner.get_network_info()
        print(f"\nベイジアンネットワーク (Layer 1):")
        print(f"  ノード数: {network_info['n_nodes']}")
        print(f"  エッジ数: {network_info['n_edges']}")
        print(f"  最大入次数: {network_info['max_indegree']}")
    
    if recommender.prob_learner:
        cond_probs = recommender.prob_learner.get_all_conditional_probs()
        print(f"\n条件付き確率 (Layer 2):")
        print(f"  L1カテゴリ数: {len(cond_probs)}")
        total_l2 = sum(len(v) for v in cond_probs.values())
        print(f"  L2カテゴリ数（総計）: {total_l2}")
    
    if recommender.mf_learner:
        mf_info = recommender.mf_learner.get_model_info()
        print(f"\n行列分解 (Layer 3):")
        print(f"  学習済みカテゴリ数: {mf_info['n_categories']}")
        print(f"  潜在因子数: {mf_info['n_components']}")
    
    # 検証結果
    print("\n" + "=" * 80)
    print("検証結果")
    print("=" * 80)
    
    checks = []
    
    # カテゴリ数の検証
    if recommender.hierarchy:
        l1_ok = 5 <= len(recommender.hierarchy.level1_categories) <= 30
        checks.append(("L1カテゴリ数（5-30個）", l1_ok))
        
        l2_ok = 10 <= len(recommender.hierarchy.level2_categories) <= 100
        checks.append(("L2カテゴリ数（10-100個）", l2_ok))
        
        l3_ok = len(recommender.hierarchy.level3_categories) >= 50
        checks.append(("L3カテゴリ数（50個以上）", l3_ok))
        
        skill_ok = 200 <= len(recommender.hierarchy.skill_to_category) <= 300
        checks.append(("スキル数（200-300個）", skill_ok))
    
    # ネットワークの検証
    if recommender.network_learner and recommender.network_learner.model:
        network_ok = recommender.network_learner.model is not None
        checks.append(("ベイジアンネットワーク学習成功", network_ok))
    
    # 条件付き確率の検証
    if recommender.prob_learner:
        prob_ok = len(recommender.prob_learner.get_all_conditional_probs()) > 0
        checks.append(("条件付き確率学習成功", prob_ok))
    
    # MFの検証
    if recommender.mf_learner:
        mf_ok = len(recommender.mf_learner.category_models) > 0
        checks.append(("行列分解学習成功", mf_ok))
    
    # 推薦生成の検証
    try:
        test_recs = recommender.recommend(sample_users[0], top_n=5)
        rec_ok = len(test_recs) > 0
        checks.append(("推薦生成成功", rec_ok))
    except:
        checks.append(("推薦生成成功", False))
    
    # 結果を表示
    for check_name, result in checks:
        status = "✅ OK" if result else "❌ NG"
        print(f"{status}: {check_name}")
    
    all_ok = all(result for _, result in checks)
    
    print("\n" + "=" * 80)
    if all_ok:
        print("✅ すべての検証に合格しました！")
    else:
        print("⚠️  一部の検証に失敗しました")
    print("=" * 80)
    
    return recommender


if __name__ == '__main__':
    recommender = main()
