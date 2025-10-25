"""
Knowledge Graph のテストスクリプト

実際のデータでグラフを構築し、基本機能をテストする
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.graph.knowledge_graph import CompetenceKnowledgeGraph


def test_knowledge_graph():
    """Knowledge Graphのテスト"""

    print("=" * 80)
    print("Knowledge Graph テスト")
    print("=" * 80)

    # 1. データ読み込み
    print("\n[1/3] データ読み込み中...")
    loader = DataLoader()
    raw_data = loader.load_all_data()

    # 2. データ変換
    print("\n[2/3] データ変換中...")
    transformer = DataTransformer()
    transformed_data = transformer.transform_all(raw_data)

    # 3. Knowledge Graph 構築
    print("\n[3/3] Knowledge Graph 構築中...")
    kg = CompetenceKnowledgeGraph(
        member_competence=transformed_data['member_competence'],
        member_master=transformed_data['members_clean'],
        competence_master=transformed_data['competence_master']
    )

    # 4. 基本機能のテスト
    print("\n" + "=" * 80)
    print("基本機能テスト")
    print("=" * 80)

    # サンプルメンバーを選択
    sample_member_code = transformed_data['members_clean']['メンバーコード'].iloc[0]
    sample_member_name = transformed_data['members_clean']['メンバー名'].iloc[0]

    print(f"\nテスト対象メンバー: {sample_member_name} ({sample_member_code})")

    # (1) メンバーの習得済み力量を取得
    acquired = kg.get_member_acquired_competences(sample_member_code)
    print(f"\n  習得済み力量数: {len(acquired)}")
    print(f"  習得済み力量（最初の5件）: {list(acquired)[:5]}")

    # (2) メンバーの類似メンバーを取得
    member_node = f"member_{sample_member_code}"
    similar_members = kg.get_neighbors(member_node, edge_type="similar")
    print(f"\n  類似メンバー数: {len(similar_members)}")
    if similar_members:
        for similar_node in similar_members[:3]:
            similar_info = kg.get_node_info(similar_node)
            edge_data = kg.G[member_node][similar_node]
            print(f"    - {similar_info.get('name', 'N/A')} (類似度: {edge_data.get('similarity', 0):.3f})")

    # (3) サンプル力量のカテゴリーを取得
    if acquired:
        sample_comp_code = list(acquired)[0]
        category = kg.get_competence_category(sample_comp_code)
        comp_info = kg.get_node_info(f"competence_{sample_comp_code}")
        print(f"\n  サンプル力量: {comp_info.get('name', 'N/A')} ({sample_comp_code})")
        print(f"    カテゴリー: {category}")
        print(f"    タイプ: {comp_info.get('type', 'N/A')}")

    # (4) グラフのエクスポート（任意）
    # kg.export_to_gexf("output/knowledge_graph.gexf")

    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)

    return kg


if __name__ == "__main__":
    kg = test_knowledge_graph()
