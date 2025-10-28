"""
Category Hierarchy Handler

カテゴリー階層構造の解析と活用
"""

import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional


logger = logging.getLogger(__name__)


class CategoryHierarchy:
    """カテゴリー階層を扱うクラス

    カテゴリー名の階層構造（例: "製造部 > 製造部共通力量"）を解析し、
    親子関係を明示的に管理する。
    """

    def __init__(self, competence_master: pd.DataFrame):
        """
        Args:
            competence_master: 力量マスタ（力量カテゴリー名列を含む）
        """
        self.competence_master = competence_master
        self.hierarchy = self._build_hierarchy()
        self.parent_map = self._build_parent_map()
        self.children_map = self._build_children_map()

        logger.info("\nCategory Hierarchy 構築完了")
        logger.info("  カテゴリー総数: %d", len(self.hierarchy))
        logger.info("  階層レベル数: %d", self._count_levels())

    def _build_hierarchy(self) -> Dict[str, Dict]:
        """カテゴリー階層を構築

        Returns:
            {カテゴリーフルパス: {'level': int, 'path': List[str], 'leaf': bool}}
        """
        hierarchy = {}

        for _, row in self.competence_master.iterrows():
            category = row.get('力量カテゴリー名')
            if pd.isna(category) or not str(category).strip():
                continue

            # 階層をパースする
            path = [p.strip() for p in str(category).split('>')]

            # 各レベルのカテゴリーを登録
            for i in range(len(path)):
                partial_path = ' > '.join(path[:i+1])

                if partial_path not in hierarchy:
                    hierarchy[partial_path] = {
                        'level': i,
                        'path': path[:i+1],
                        'leaf': False,
                        'full_path': partial_path
                    }

            # 最下層は葉ノード
            if path:
                full_path = ' > '.join(path)
                if full_path in hierarchy:
                    hierarchy[full_path]['leaf'] = True

        return hierarchy

    def _build_parent_map(self) -> Dict[str, Optional[str]]:
        """親カテゴリーマッピングを構築

        Returns:
            {カテゴリー: 親カテゴリー}
        """
        parent_map = {}

        for category, info in self.hierarchy.items():
            if info['level'] == 0:
                parent_map[category] = None
            else:
                path = info['path']
                parent_path = ' > '.join(path[:-1])
                parent_map[category] = parent_path

        return parent_map

    def _build_children_map(self) -> Dict[str, List[str]]:
        """子カテゴリーマッピングを構築

        Returns:
            {カテゴリー: [子カテゴリーリスト]}
        """
        children_map = {}

        for category in self.hierarchy.keys():
            children_map[category] = []

        for category, parent in self.parent_map.items():
            if parent is not None:
                children_map[parent].append(category)

        return children_map

    def get_parent(self, category: str) -> Optional[str]:
        """親カテゴリーを取得"""
        return self.parent_map.get(category)

    def get_children(self, category: str) -> List[str]:
        """子カテゴリーを取得"""
        return self.children_map.get(category, [])

    def get_ancestors(self, category: str) -> List[str]:
        """祖先カテゴリーを全て取得（親、祖父、...）

        Returns:
            [親, 祖父, 曾祖父, ...] の順
        """
        ancestors = []
        current = category

        while True:
            parent = self.get_parent(current)
            if parent is None:
                break
            ancestors.append(parent)
            current = parent

        return ancestors

    def get_descendants(self, category: str) -> List[str]:
        """子孫カテゴリーを全て取得（子、孫、...）

        Returns:
            全ての子孫のリスト
        """
        descendants = []

        def _collect_descendants(cat):
            children = self.get_children(cat)
            for child in children:
                descendants.append(child)
                _collect_descendants(child)

        _collect_descendants(category)
        return descendants

    def get_siblings(self, category: str) -> List[str]:
        """兄弟カテゴリーを取得（同じ親を持つカテゴリー）"""
        parent = self.get_parent(category)
        if parent is None:
            # ルートレベルの兄弟を取得
            return [c for c in self.hierarchy.keys()
                    if self.hierarchy[c]['level'] == 0 and c != category]

        siblings = self.get_children(parent)
        return [s for s in siblings if s != category]

    def get_level(self, category: str) -> int:
        """カテゴリーの階層レベルを取得（0=ルート）"""
        info = self.hierarchy.get(category)
        return info['level'] if info else -1

    def is_leaf(self, category: str) -> bool:
        """葉ノード（最下層）かどうか"""
        info = self.hierarchy.get(category)
        return info['leaf'] if info else False

    def _count_levels(self) -> int:
        """階層の深さを取得"""
        if not self.hierarchy:
            return 0
        return max(info['level'] for info in self.hierarchy.values()) + 1

    def get_category_tree(self) -> Dict:
        """カテゴリーツリーを取得（可視化用）

        Returns:
            ネストされた辞書形式のツリー
        """
        tree = {}

        # ルートレベルから構築
        roots = [cat for cat, info in self.hierarchy.items() if info['level'] == 0]

        def _build_tree(category):
            children = self.get_children(category)
            if not children:
                return None

            subtree = {}
            for child in children:
                subtree[child] = _build_tree(child)
            return subtree

        for root in roots:
            tree[root] = _build_tree(root)

        return tree

    def get_common_ancestor(self, category1: str, category2: str) -> Optional[str]:
        """2つのカテゴリーの最も近い共通祖先を取得

        Returns:
            共通祖先カテゴリー（存在しない場合はNone）
        """
        ancestors1 = set([category1] + self.get_ancestors(category1))
        ancestors2 = set([category2] + self.get_ancestors(category2))

        common = ancestors1 & ancestors2

        if not common:
            return None

        # 最も深い（レベルが大きい）共通祖先を返す
        return max(common, key=lambda c: self.get_level(c))

    def calculate_category_distance(self, category1: str, category2: str) -> int:
        """2つのカテゴリー間の距離を計算

        距離 = カテゴリー1から共通祖先までの距離 + カテゴリー2から共通祖先までの距離

        Returns:
            距離（同じカテゴリーなら0、共通祖先がない場合は-1）
        """
        if category1 == category2:
            return 0

        common_ancestor = self.get_common_ancestor(category1, category2)
        if common_ancestor is None:
            return -1

        # 各カテゴリーから共通祖先までの距離
        dist1 = self.get_level(category1) - self.get_level(common_ancestor)
        dist2 = self.get_level(category2) - self.get_level(common_ancestor)

        return dist1 + dist2
