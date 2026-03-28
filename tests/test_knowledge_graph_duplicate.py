"""
Knowledge Graph のテストスクリプト

実際のデータでグラフを構築し、基本機能をテストする
"""

import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.graph.knowledge_graph import CompetenceKnowledgeGraph


logger = logging.getLogger(__name__)


def test_knowledge_graph():
    """Knowledge Graphのテスト"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("Knowledge Graph テスト")
    logger.info("=" * 80)

    # 1. データ読み込み
    logger.info("\n[1/3] データ読み込み中...")
    loader = DataLoader()
    raw_data = loader.load_all_data()

    # 2. データ変換
    logger.info("\n[2/3] データ変換中...")
    transformer = DataTransformer()
    transformed_data = transformer.transform_all(raw_data)

    # 3. Knowledge Graph 構築
    logger.info("\n[3/3] Knowledge Graph 構築中...")
    kg = CompetenceKnowledgeGraph(
        member_competence=transformed_data["member_competence"],
        member_master=transformed_data["members_clean"],
        competence_master=transformed_data["competence_master"],
    )

    # 4. 基本機能のテスト
    logger.info("\n" + "=" * 80)
    logger.info("基本機能テスト")
    logger.info("=" * 80)

    # サンプルメンバーを選択
    sample_member_code = transformed_data["members_clean"]["メンバーコード"].iloc[0]
    sample_member_name = transformed_data["members_clean"]["メンバー名"].iloc[0]

    logger.info("\nテスト対象メンバー: %s (%s)", sample_member_name, sample_member_code)

    # (1) メンバーの習得済み力量を取得
    acquired = kg.get_member_acquired_competences(sample_member_code)
    logger.info("\n  習得済み力量数: %d", len(acquired))
    logger.info("  習得済み力量（最初の5件）: %s", list(acquired)[:5])

    # (2) メンバーの類似メンバーを取得
    member_node = f"member_{sample_member_code}"
    similar_members = kg.get_neighbors(member_node, edge_type="similar")
    logger.info("\n  類似メンバー数: %d", len(similar_members))
    if similar_members:
        for similar_node in similar_members[:3]:
            similar_info = kg.get_node_info(similar_node)
            edge_data = kg.G[member_node][similar_node]
            logger.info(
                "    - %s (類似度: %.3f)",
                similar_info.get("name", "N/A"),
                edge_data.get("similarity", 0),
            )

    # (3) サンプル力量のカテゴリーを取得
    if acquired:
        sample_comp_code = list(acquired)[0]
        category = kg.get_competence_category(sample_comp_code)
        comp_info = kg.get_node_info(f"competence_{sample_comp_code}")
        logger.info("\n  サンプル力量: %s (%s)", comp_info.get("name", "N/A"), sample_comp_code)
        logger.info("    カテゴリー: %s", category)
        logger.info("    タイプ: %s", comp_info.get("type", "N/A"))

    # (4) グラフのエクスポート（任意）
    # kg.export_to_gexf("output/knowledge_graph.gexf")

    logger.info("\n" + "=" * 80)
    logger.info("テスト完了")
    logger.info("=" * 80)

    return kg


if __name__ == "__main__":
    kg = test_knowledge_graph()
