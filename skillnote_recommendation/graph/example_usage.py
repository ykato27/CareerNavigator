"""
Enhanced Graph-based Recommendation: 使用例

改善されたグラフベース推薦システムの使い方を示すサンプルコード
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# 改善されたモジュールのインポート
from skillnote_recommendation.ml.enhanced_graph_recommender import (
    EnhancedSkillTransitionGraphRecommender,
)
from skillnote_recommendation.graph.enhanced_path_visualizer import (
    EnhancedPathVisualizer,
    EdgeStatistics,
    create_comparison_view,
)
from skillnote_recommendation.graph.sankey_visualizer import (
    SkillTransitionSankeyVisualizer,
    TimeBasedSankeyVisualizer,
)


def example_1_basic_usage():
    """例1: 基本的な使い方"""
    print("=" * 80)
    print("例1: 基本的な使い方")
    print("=" * 80)

    # サンプルデータの準備
    member_competence = pd.DataFrame(
        {
            "メンバーコード": ["M001", "M001", "M001", "M002", "M002"],
            "力量コード": ["S001", "S002", "S003", "S001", "S003"],
            "取得日": ["2024-01-01", "2024-02-15", "2024-04-01", "2024-01-10", "2024-03-20"],
        }
    )

    competence_master = pd.DataFrame(
        {
            "力量コード": ["S001", "S002", "S003"],
            "力量名": ["Python基礎", "データ分析", "機械学習"],
        }
    )

    # 推薦システムの初期化
    recommender = EnhancedSkillTransitionGraphRecommender(
        time_window_days=180,
        min_transition_count=1,
        time_decay_factor=0.01,
        use_time_decay=True,
        use_robust_scaling=True,
    )

    # 学習
    recommender.fit(member_competence, competence_master)

    # 推薦
    recommendations = recommender.recommend(member_code="M001", n=5, exclude_acquired=True)

    # 結果表示
    print("\n推薦結果:")
    for rec in recommendations:
        print(
            f"  {rec.rank}. {rec.skill_name} (スコア: {rec.score:.2f}, 信頼度: {rec.confidence:.2f})"
        )
        print(f"     理由: {rec.explanation}")
        print()


def example_2_enhanced_visualization():
    """例2: 高度な可視化"""
    print("=" * 80)
    print("例2: 高度な可視化（Enhanced Path Visualizer）")
    print("=" * 80)

    # サンプルパスデータ
    paths = [
        [
            {"id": "member_M001", "type": "member", "name": "山田太郎"},
            {"id": "competence_S001", "type": "competence", "name": "Python基礎"},
            {"id": "category_Programming", "type": "category", "name": "プログラミング"},
            {"id": "competence_S002", "type": "competence", "name": "データ分析"},
        ],
        [
            {"id": "member_M001", "type": "member", "name": "山田太郎"},
            {"id": "competence_S001", "type": "competence", "name": "Python基礎"},
            {"id": "member_M002", "type": "similar_member", "name": "田中次郎"},
            {"id": "competence_S002", "type": "competence", "name": "データ分析"},
        ],
    ]

    # エッジ統計情報
    edge_statistics = {
        ("competence_S001", "category_Programming"): EdgeStatistics(
            source_name="Python基礎",
            target_name="プログラミング",
            transition_count=15,
            avg_days=30.0,
            median_days=28.0,
            success_rate=0.85,
        ),
        ("category_Programming", "competence_S002"): EdgeStatistics(
            source_name="プログラミング",
            target_name="データ分析",
            transition_count=12,
            avg_days=45.0,
            median_days=40.0,
            success_rate=0.90,
        ),
    }

    # 可視化の作成
    visualizer = EnhancedPathVisualizer(
        layout_algorithm="fruchterman_reingold", colorblind_safe=True, show_edge_statistics=True
    )

    fig = visualizer.visualize_paths(
        paths=paths,
        target_member_name="山田太郎",
        target_competence_name="データ分析",
        edge_statistics=edge_statistics,
        path_scores=[0.9, 0.7],
        min_quality_score=0.0,
    )

    # 表示（Streamlitの場合は st.plotly_chart(fig)）
    print("可視化を生成しました")
    # fig.show()  # ブラウザで表示する場合はコメントアウトを解除

    return fig


def example_3_sankey_diagram():
    """例3: サンキーダイアグラム"""
    print("=" * 80)
    print("例3: サンキーダイアグラム")
    print("=" * 80)

    # サンプルパスデータ
    paths = [
        [
            {"id": "member_M001", "type": "member", "name": "山田太郎"},
            {"id": "competence_S001", "type": "competence", "name": "Python基礎"},
            {"id": "competence_S002", "type": "competence", "name": "データ分析"},
        ],
        [
            {"id": "member_M001", "type": "member", "name": "山田太郎"},
            {"id": "competence_S001", "type": "competence", "name": "Python基礎"},
            {"id": "category_Programming", "type": "category", "name": "プログラミング"},
            {"id": "competence_S002", "type": "competence", "name": "データ分析"},
        ],
    ]

    # 遷移カウント
    transition_counts = {
        ("member_M001", "competence_S001"): 1,
        ("competence_S001", "competence_S002"): 10,
        ("competence_S001", "category_Programming"): 8,
        ("category_Programming", "competence_S002"): 8,
    }

    # サンキーダイアグラムの作成
    sankey_vis = SkillTransitionSankeyVisualizer(
        show_percentages=True, color_by_category=True, min_flow_threshold=1
    )

    fig = sankey_vis.visualize_transition_flow(
        paths=paths,
        target_member_name="山田太郎",
        target_competence_name="データ分析",
        transition_counts=transition_counts,
    )

    print("サンキーダイアグラムを生成しました")
    # fig.show()

    return fig


def example_4_time_based_visualization():
    """例4: 時間情報を含む可視化"""
    print("=" * 80)
    print("例4: 時間情報を含むサンキーダイアグラム")
    print("=" * 80)

    # サンプルパスデータ
    paths = [
        [
            {"id": "member_M001", "type": "member", "name": "山田太郎"},
            {"id": "competence_S001", "type": "competence", "name": "Python基礎"},
            {"id": "competence_S002", "type": "competence", "name": "データ分析"},
        ],
    ]

    # 時間情報
    edge_time_info = {
        ("member_M001", "competence_S001"): {"avg_days": 0, "median_days": 0, "count": 1},
        ("competence_S001", "competence_S002"): {
            "avg_days": 45.0,
            "median_days": 40.0,
            "count": 10,
        },
    }

    # 時間考慮サンキーダイアグラムの作成
    time_sankey = TimeBasedSankeyVisualizer()

    fig = time_sankey.visualize_with_time_info(
        paths=paths,
        target_member_name="山田太郎",
        target_competence_name="データ分析",
        edge_time_info=edge_time_info,
    )

    print("時間考慮サンキーダイアグラムを生成しました")
    print("色の意味:")
    print("  🟢 緑: 0-30日（速い）")
    print("  🟡 黄: 30-90日（普通）")
    print("  🟠 オレンジ: 90-180日（遅い）")
    print("  🔴 赤: 180日+（とても遅い）")
    # fig.show()

    return fig


def example_5_comparison_view():
    """例5: 複数の可視化を比較"""
    print("=" * 80)
    print("例5: 複数の可視化を並べて比較")
    print("=" * 80)

    # 各可視化を生成
    fig_enhanced = example_2_enhanced_visualization()
    fig_sankey = example_3_sankey_diagram()
    fig_time = example_4_time_based_visualization()

    # 比較ビューを作成
    fig_comparison = create_comparison_view(
        [
            ("拡張パス可視化", fig_enhanced),
            ("サンキーダイアグラム", fig_sankey),
            ("時間考慮サンキー", fig_time),
        ]
    )

    print("比較ビューを生成しました")
    # fig_comparison.show()

    return fig_comparison


def example_6_advanced_features():
    """例6: 高度な機能の活用"""
    print("=" * 80)
    print("例6: 高度な機能の活用")
    print("=" * 80)

    # サンプルデータの準備
    member_competence = pd.DataFrame(
        {
            "メンバーコード": ["M001"] * 10 + ["M002"] * 8,
            "力量コード": [
                "S001",
                "S002",
                "S003",
                "S004",
                "S005",
                "S006",
                "S007",
                "S008",
                "S009",
                "S010",
            ]
            + ["S001", "S002", "S003", "S005", "S007", "S008", "S009", "S010"],
            "取得日": [
                "2023-01-01",
                "2023-02-15",
                "2023-04-01",
                "2023-06-01",
                "2023-08-01",
                "2023-10-01",
                "2023-12-01",
                "2024-02-01",
                "2024-04-01",
                "2024-06-01",
            ]
            + [
                "2023-02-01",
                "2023-03-15",
                "2023-05-01",
                "2023-09-01",
                "2023-11-01",
                "2024-01-01",
                "2024-03-01",
                "2024-05-01",
            ],
        }
    )

    competence_master = pd.DataFrame(
        {
            "力量コード": [f"S{i:03d}" for i in range(1, 11)],
            "力量名": [f"スキル{i}" for i in range(1, 11)],
        }
    )

    # 推薦システムの初期化（高度な設定）
    recommender = EnhancedSkillTransitionGraphRecommender(
        time_window_days=365,
        min_transition_count=1,
        embedding_dim=128,
        walk_length=15,
        num_walks=100,
        p=0.5,  # DFS寄り（より局所的な探索）
        q=2.0,  # global探索
        time_decay_factor=0.02,  # より強い時間減衰
        use_time_decay=True,
        path_quality_weight=0.4,  # パス品質を重視
        use_robust_scaling=True,
    )

    # 学習
    print("\n推薦システムを学習中...")
    recommender.fit(member_competence, competence_master)

    # 推薦
    print("\n推薦を生成中...")
    recommendations = recommender.recommend(member_code="M001", n=10, exclude_acquired=True)

    # 結果表示
    print("\n推薦結果:")
    for rec in recommendations:
        print(f"  {rec.rank}. {rec.skill_name}")
        print(f"     スコア: {rec.score:.3f} | 信頼度: {rec.confidence:.3f}")
        print(f"     理由: {rec.explanation}")
        print()

    # エッジ統計情報を取得
    edge_stats = recommender.get_edge_statistics()
    print(f"\nエッジ統計情報: {len(edge_stats)}個のエッジ")

    # 統計情報の例
    if edge_stats:
        sample_edge = list(edge_stats.items())[0]
        print(f"\nサンプルエッジ: {sample_edge[0]}")
        print(f"  詳細: {sample_edge[1]}")


def example_7_heatmap():
    """例7: ヒートマップで全体像を把握"""
    print("=" * 80)
    print("例7: ヒートマップでスキル遷移パターンを俯瞰")
    print("=" * 80)

    # 遷移マトリックス（サンプル）
    transition_matrix = {
        ("S001", "S002"): 15,
        ("S001", "S003"): 8,
        ("S002", "S003"): 12,
        ("S002", "S004"): 5,
        ("S003", "S004"): 10,
        ("S003", "S005"): 7,
    }

    skill_names = {
        "S001": "Python基礎",
        "S002": "データ分析",
        "S003": "機械学習",
        "S004": "深層学習",
        "S005": "MLOps",
    }

    # ヒートマップの作成
    sankey_vis = SkillTransitionSankeyVisualizer()

    fig = sankey_vis.visualize_skill_matrix_heatmap(
        transition_matrix=transition_matrix, skill_names=skill_names
    )

    print("ヒートマップを生成しました")
    print("全体のスキル遷移パターンを俯瞰できます")
    # fig.show()

    return fig


def main():
    """全ての例を実行"""
    print("\n" + "=" * 80)
    print("Enhanced Graph-based Recommendation: 使用例")
    print("=" * 80 + "\n")

    # 例1: 基本的な使い方
    example_1_basic_usage()
    print("\n")

    # 例2: 高度な可視化
    example_2_enhanced_visualization()
    print("\n")

    # 例3: サンキーダイアグラム
    example_3_sankey_diagram()
    print("\n")

    # 例4: 時間情報を含む可視化
    example_4_time_based_visualization()
    print("\n")

    # 例5: 複数の可視化を比較
    # example_5_comparison_view()
    # print("\n")

    # 例6: 高度な機能の活用
    example_6_advanced_features()
    print("\n")

    # 例7: ヒートマップ
    example_7_heatmap()
    print("\n")

    print("=" * 80)
    print("全ての例を実行しました")
    print("=" * 80)


if __name__ == "__main__":
    main()
