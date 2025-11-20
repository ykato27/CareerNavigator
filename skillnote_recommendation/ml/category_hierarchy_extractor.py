"""
カテゴリ階層抽出モジュール

CSVファイルから3レベルのカテゴリ階層（L1: 大カテゴリ、L2: 中カテゴリ、L3: 小カテゴリ）を抽出し、
スキルとカテゴリのマッピングを構築します。
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)



@dataclass
class CategoryHierarchy:
    """カテゴリ階層を表すデータクラス"""
    
    # カテゴリコード -> カテゴリ名のマッピング
    category_names: Dict[str, str] = field(default_factory=dict)
    
    # レベル別カテゴリコードのリスト
    level1_categories: List[str] = field(default_factory=list)  # 大カテゴリ
    level2_categories: List[str] = field(default_factory=list)  # 中カテゴリ
    level3_categories: List[str] = field(default_factory=list)  # 小カテゴリ
    
    # 親子関係のマッピング
    # child_code -> parent_code
    parent_mapping: Dict[str, str] = field(default_factory=dict)
    
    # 子のリスト
    # parent_code -> [child_codes]
    children_mapping: Dict[str, List[str]] = field(default_factory=dict)
    
    # スキルコード -> カテゴリコード（最も詳細なレベル）のマッピング
    skill_to_category: Dict[str, str] = field(default_factory=dict)
    
    # カテゴリコード -> スキルコードのリスト
    category_to_skills: Dict[str, List[str]] = field(default_factory=dict)
    
    def get_level(self, category_code: str) -> int:
        """カテゴリのレベルを取得（1, 2, 3）"""
        if category_code in self.level1_categories:
            return 1
        elif category_code in self.level2_categories:
            return 2
        elif category_code in self.level3_categories:
            return 3
        else:
            return 0
    
    def get_ancestors(self, category_code: str) -> List[str]:
        """カテゴリの祖先（親、祖父母など）をリストで取得"""
        ancestors = []
        current = category_code
        while current in self.parent_mapping:
            parent = self.parent_mapping[current]
            ancestors.append(parent)
            current = parent
        return ancestors
    
    def get_l1_category(self, category_code: str) -> Optional[str]:
        """カテゴリのL1（大カテゴリ）を取得"""
        if category_code in self.level1_categories:
            return category_code
        ancestors = self.get_ancestors(category_code)
        # 祖先の中でL1に属するものを探す
        for ancestor in ancestors:
            if ancestor in self.level1_categories:
                return ancestor
        return None
    
    def get_l2_category(self, category_code: str) -> Optional[str]:
        """カテゴリのL2（中カテゴリ）を取得"""
        if category_code in self.level2_categories:
            return category_code
        # L3の場合、親がL2のはず
        if category_code in self.level3_categories:
            parent = self.parent_mapping.get(category_code)
            if parent and parent in self.level2_categories:
                return parent
        return None


class CategoryHierarchyExtractor:
    """カテゴリ階層抽出クラス"""
    
    def __init__(
        self,
        category_csv_path: str = None,
        skill_csv_path: str = None,
        category_df: pd.DataFrame = None,
        skill_df: pd.DataFrame = None
    ):
        """
        初期化

        Args:
            category_csv_path: カテゴリマスタCSVのパス（オプション）
            skill_csv_path: スキルマスタCSVのパス（オプション）
            category_df: カテゴリマスタDataFrame（オプション）
            skill_df: スキルマスタDataFrame（オプション）
        """
        self.category_csv_path = category_csv_path
        self.skill_csv_path = skill_csv_path
        self.category_df = category_df
        self.skill_df = skill_df
        self.hierarchy: Optional[CategoryHierarchy] = None
    
    def extract_hierarchy(self) -> CategoryHierarchy:
        """
        カテゴリ階層を抽出

        Returns:
            CategoryHierarchy: 抽出された階層構造
        """
        logger.info("カテゴリ階層の抽出を開始")

        # カテゴリマスタを取得（DataFrameまたはCSVから）
        if self.category_df is not None:
            category_df = self.category_df
            logger.info(f"カテゴリマスタ（DataFrame）: {len(category_df)}行")
        elif self.category_csv_path:
            category_df = pd.read_csv(self.category_csv_path)
            logger.info(f"カテゴリマスタ読み込み: {len(category_df)}行")
        else:
            raise ValueError("category_df または category_csv_path のいずれかが必要です")

        # スキルマスタを取得（DataFrameまたはCSVから）
        if self.skill_df is not None:
            skill_df = self.skill_df
            logger.info(f"スキルマスタ（DataFrame）: {len(skill_df)}行")
        elif self.skill_csv_path:
            skill_df = pd.read_csv(self.skill_csv_path)
            logger.info(f"スキルマスタ読み込み: {len(skill_df)}行")
        else:
            raise ValueError("skill_df または skill_csv_path のいずれかが必要です")
        
        # 階層構造を構築
        hierarchy = CategoryHierarchy()
        
        # カテゴリ階層を解析
        self._parse_category_hierarchy(category_df, hierarchy)
        
        # スキルとカテゴリのマッピングを構築
        self._build_skill_category_mapping(skill_df, hierarchy)
        
        # 統計情報をログ出力
        logger.info(
            f"階層抽出完了: L1={len(hierarchy.level1_categories)}, "
            f"L2={len(hierarchy.level2_categories)}, "
            f"L3={len(hierarchy.level3_categories)}, "
            f"スキル={len(hierarchy.skill_to_category)}"
        )
        
        self.hierarchy = hierarchy
        return hierarchy
    
    def _parse_category_hierarchy(
        self,
        category_df: pd.DataFrame,
        hierarchy: CategoryHierarchy
    ):
        """
        カテゴリマスタから階層構造を解析
        
        Args:
            category_df: カテゴリマスタのDataFrame
            hierarchy: 構築中のCategoryHierarchy
        """
        # カテゴリ名のカラムをインデックス位置で特定（重複カラム名に対応）
        name_column_indices = [i for i, col in enumerate(category_df.columns)
                              if '力量カテゴリー名' in str(col)]

        # カテゴリコードのカラムを特定（マーカーあり/なし両方に対応）
        category_code_col_idx = None
        for i, col in enumerate(category_df.columns):
            if '力量カテゴリーコード' in str(col):
                category_code_col_idx = i
                break

        if category_code_col_idx is None:
            raise ValueError("力量カテゴリーコードカラムが見つかりません")

        logger.info(f"カテゴリ名カラム数: {len(name_column_indices)}")
        logger.info(f"カテゴリ名カラムインデックス: {name_column_indices}")
        logger.info(f"カテゴリコードカラムインデックス: {category_code_col_idx}")

        # 重複カラム名のチェック
        duplicate_cols = category_df.columns[category_df.columns.duplicated()].tolist()
        if duplicate_cols:
            logger.warning(f"警告: DataFrameに重複カラムがあります（インデックスでアクセスします）: {len(set(duplicate_cols))}種類")

        # 各行を処理
        for _, row in category_df.iterrows():
            # カテゴリコードを位置インデックスで取得
            category_code = row.iloc[category_code_col_idx]

            # 欠損値チェック
            if pd.isna(category_code):
                continue

            # 各レベルのカテゴリ名を取得（空でない最も深いレベルを特定）
            category_names = []
            for col_idx in name_column_indices:
                name = row.iloc[col_idx]

                # スカラー値として処理
                if pd.notna(name) and str(name).strip():
                    category_names.append(str(name).strip())

            if not category_names:
                continue

            # カテゴリのレベルを判定（非空のカラム数）
            level = len(category_names)
            
            # カテゴリ名を結合（階層パス）
            full_name = ' > '.join(category_names)
            hierarchy.category_names[category_code] = full_name
            
            # レベル別に分類
            if level == 1:
                hierarchy.level1_categories.append(category_code)
            elif level == 2:
                hierarchy.level2_categories.append(category_code)
            elif level >= 3:
                # レベル3以上は全てL3として扱う
                hierarchy.level3_categories.append(category_code)
        
        # 親子関係を構築（カテゴリコードの接頭辞ベース）
        self._build_parent_child_relationships(hierarchy)
        
        logger.info(
            f"カテゴリ階層解析完了: L1={len(hierarchy.level1_categories)}, "
            f"L2={len(hierarchy.level2_categories)}, "
            f"L3={len(hierarchy.level3_categories)}"
        )
    
    def _build_parent_child_relationships(self, hierarchy: CategoryHierarchy):
        """
        カテゴリコードの接頭辞に基づいて親子関係を構築
        
        カテゴリコードの形式:
        - L1: CTG100000000 (最初の3桁が100)
        - L2: CTG101000000 (最初の6桁が101000)
        - L3: CTG101010000 (最初の9桁が101010000)
        
        Args:
            hierarchy: 構築中のCategoryHierarchy
        """
        all_categories = (
            hierarchy.level1_categories + 
            hierarchy.level2_categories + 
            hierarchy.level3_categories
        )
        
        # 各カテゴリについて親を探す
        for category_code in all_categories:
            # カテゴリコードから数値部分を抽出
            if not category_code.startswith('CTG'):
                continue
            
            code_num = category_code[3:]  # 'CTG'を除去
            
            # 親候補を探す（より短い接頭辞を持つカテゴリ）
            potential_parents = []
            for other_code in all_categories:
                if other_code == category_code:
                    continue
                if not other_code.startswith('CTG'):
                    continue
                
                other_num = other_code[3:]
                # category_codeがother_codeの接頭辞で始まり、かつより長い場合
                if code_num.startswith(other_num) and len(code_num) > len(other_num):
                    # ゼロでない部分の長さで親の近さを判定
                    non_zero_len = len(other_num.rstrip('0'))
                    potential_parents.append((other_code, non_zero_len))
            
            # 最も近い親（最も長い接頭辞）を選択
            if potential_parents:
                potential_parents.sort(key=lambda x: x[1], reverse=True)
                parent_code = potential_parents[0][0]
                hierarchy.parent_mapping[category_code] = parent_code
                
                # 子リストに追加
                if parent_code not in hierarchy.children_mapping:
                    hierarchy.children_mapping[parent_code] = []
                hierarchy.children_mapping[parent_code].append(category_code)
        
        logger.info(f"親子関係構築完了: {len(hierarchy.parent_mapping)}個の親子関係")
    
    def _build_skill_category_mapping(
        self,
        skill_df: pd.DataFrame,
        hierarchy: CategoryHierarchy
    ):
        """
        スキルとカテゴリのマッピングを構築
        
        Args:
            skill_df: スキルマスタのDataFrame
            hierarchy: 構築中のCategoryHierarchy
        """
        # カラム名を柔軟に取得（マーカーあり/なし両方に対応）
        competence_type_col = None
        skill_code_col = None
        category_code_col = None

        for col in skill_df.columns:
            if '力量タイプ' in col:
                competence_type_col = col
            elif '力量コード' in col and 'カテゴリー' not in col:
                skill_code_col = col
            elif '力量カテゴリーコード' in col:
                category_code_col = col

        if competence_type_col is None:
            raise ValueError("力量タイプカラムが見つかりません")
        if skill_code_col is None:
            raise ValueError("力量コードカラムが見つかりません")
        if category_code_col is None:
            raise ValueError("力量カテゴリーコードカラムが見つかりません")

        logger.info(f"スキルマスタカラム: 力量タイプ={competence_type_col}, 力量コード={skill_code_col}, カテゴリコード={category_code_col}")

        # 重複カラム名のチェック
        duplicate_skill_cols = skill_df.columns[skill_df.columns.duplicated()].tolist()
        if duplicate_skill_cols:
            logger.warning(f"警告: スキルマスタDataFrameに重複カラムがあります: {duplicate_skill_cols}")

        # 力量タイプがSKILLのものだけを対象
        skill_rows = skill_df[skill_df[competence_type_col] == 'SKILL']
        logger.info(f"SKILL行数: {len(skill_rows)}")

        for _, row in skill_rows.iterrows():
            skill_code = row[skill_code_col]
            category_code = row[category_code_col]

            # Series の場合は最初の値を取得
            if isinstance(skill_code, pd.Series):
                if not skill_code.empty:
                    skill_code = skill_code.iloc[0]
                else:
                    continue

            if isinstance(category_code, pd.Series):
                if not category_code.empty:
                    category_code = category_code.iloc[0]
                else:
                    continue

            # スキル -> カテゴリのマッピング
            hierarchy.skill_to_category[skill_code] = category_code
            
            # カテゴリ -> スキルのマッピング
            if category_code not in hierarchy.category_to_skills:
                hierarchy.category_to_skills[category_code] = []
            hierarchy.category_to_skills[category_code].append(skill_code)
        
        logger.info(
            f"スキル-カテゴリマッピング構築完了: "
            f"{len(hierarchy.skill_to_category)}スキル"
        )
    
    def get_skills_by_category(
        self,
        category_code: str,
        include_descendants: bool = False
    ) -> List[str]:
        """
        カテゴリに属するスキルのリストを取得
        
        Args:
            category_code: カテゴリコード
            include_descendants: 子孫カテゴリのスキルも含めるか
            
        Returns:
            スキルコードのリスト
        """
        if self.hierarchy is None:
            raise ValueError("extract_hierarchy()を先に実行してください")
        
        skills = set()
        
        # 直接このカテゴリに属するスキル
        if category_code in self.hierarchy.category_to_skills:
            skills.update(self.hierarchy.category_to_skills[category_code])
        
        # 子孫カテゴリのスキルも含める場合
        if include_descendants:
            descendants = self._get_all_descendants(category_code)
            for desc_code in descendants:
                if desc_code in self.hierarchy.category_to_skills:
                    skills.update(self.hierarchy.category_to_skills[desc_code])
        
        return list(skills)
    
    def _get_all_descendants(self, category_code: str) -> List[str]:
        """カテゴリの全子孫を再帰的に取得"""
        if self.hierarchy is None:
            return []
        
        descendants = []
        if category_code in self.hierarchy.children_mapping:
            for child in self.hierarchy.children_mapping[category_code]:
                descendants.append(child)
                descendants.extend(self._get_all_descendants(child))
        
        return descendants
