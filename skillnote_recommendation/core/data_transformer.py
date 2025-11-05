"""
データ変換

スキルノートのデータを推薦システム用に変換
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from skillnote_recommendation.core.config import Config


logger = logging.getLogger(__name__)


class DataTransformer:
    """データ変換クラス"""
    
    @staticmethod
    def normalize_level(level_value, competence_type: str) -> int:
        """
        レベル値を正規化
        
        Args:
            level_value: 元のレベル値
            competence_type: 力量タイプ（SKILL/EDUCATION/LICENSE）
            
        Returns:
            正規化されたレベル（0-5）
        """
        if competence_type == 'SKILL':
            try:
                return int(level_value)
            except:
                return 0
        else:  # EDUCATION or LICENSE
            if pd.isna(level_value) or str(level_value).strip() == '':
                return 0
            elif str(level_value).strip() == '●':
                return 1
            else:
                return 0
    
    def create_competence_master(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        統合力量マスタを作成
        
        Args:
            data: 読み込んだデータの辞書
            
        Returns:
            統合力量マスタ
        """
        logger.info("\n" + "=" * 80)
        logger.info("統合力量マスタ作成")
        logger.info("=" * 80)
        
        # 各力量タイプのマスタを整形
        skills = data['skills'][['力量コード', '力量名', '力量カテゴリーコード', '概要']].copy()
        skills['力量タイプ'] = 'SKILL'
        skills['レベル範囲'] = '1-5'
        
        education = data['education'][['力量コード', '力量名', '力量カテゴリーコード', '概要']].copy()
        education['力量タイプ'] = 'EDUCATION'
        education['レベル範囲'] = '●'
        
        license_data = data['license'][['力量コード', '力量名', '力量カテゴリーコード', '概要']].copy()
        license_data['力量タイプ'] = 'LICENSE'
        license_data['レベル範囲'] = '●'
        
        # 統合
        competence_master = pd.concat([skills, education, license_data], ignore_index=True)
        
        # 主カテゴリコードを抽出
        competence_master['力量カテゴリーコード_主'] = competence_master['力量カテゴリーコード'].apply(
            lambda x: str(x).split(',')[0].strip() if pd.notna(x) else ''
        )
        
        # カテゴリ名マッピングを作成
        category_names = self._create_category_names(data['categories'])
        competence_master['力量カテゴリー名'] = competence_master['力量カテゴリーコード_主'].map(category_names)
        
        logger.info("\n統合完了: %d件", len(competence_master))
        logger.info(
            "  - SKILL: %d件",
            len(competence_master[competence_master['力量タイプ'] == 'SKILL'])
        )
        logger.info(
            "  - EDUCATION: %d件",
            len(competence_master[competence_master['力量タイプ'] == 'EDUCATION'])
        )
        logger.info(
            "  - LICENSE: %d件",
            len(competence_master[competence_master['力量タイプ'] == 'LICENSE'])
        )
        
        return competence_master
    
    def _create_category_names(self, df_categories: pd.DataFrame) -> dict:
        """
        カテゴリコードからカテゴリ名へのマッピングを作成
        
        Args:
            df_categories: カテゴリマスタ
            
        Returns:
            カテゴリ名マッピング辞書
        """
        category_names = {}
        
        for _, row in df_categories.iterrows():
            code = row.iloc[0]
            names = []
            for i in range(1, len(row)):
                val = row.iloc[i]
                if not pd.isna(val) and str(val).strip() != '':
                    names.append(str(val))
            category_names[code] = ' > '.join(names) if names else ''
        
        return category_names
    
    def create_member_competence(self, data: Dict[str, pd.DataFrame],
                                 competence_master: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        メンバー習得力量データを作成

        Args:
            data: 読み込んだデータの辞書
            competence_master: 統合力量マスタ

        Returns:
            (メンバー習得力量データ, 有効なメンバーコードリスト)
        """
        logger.info("\n" + "=" * 80)
        logger.info("メンバー習得力量データ作成")
        logger.info("=" * 80)

        # 習得力量データの確認
        acquired_df = data['acquired']
        logger.info("\n保有力量データ:")
        logger.info("  総行数: %d", len(acquired_df))
        logger.info("  カラム: %s", list(acquired_df.columns))

        # 必須カラムの確認
        required_columns = ['メンバーコード', '力量コード', '力量タイプ', 'レベル']
        missing_columns = [col for col in required_columns if col not in acquired_df.columns]
        if missing_columns:
            logger.error("  ⚠ 必須カラムが不足しています: %s", missing_columns)
            logger.error("  利用可能なカラム: %s", list(acquired_df.columns))
            raise ValueError(f"保有力量データに必須カラムが不足しています: {missing_columns}")

        # 有効なメンバーを抽出
        valid_members = data['members'][
            (~data['members']['メンバー名'].str.contains('削除|テスト|test', case=False, na=False)) &
            (data['members']['メンバーコード'].notna())
        ]['メンバーコード'].unique()

        logger.info("\n有効なメンバー数: %d名", len(valid_members))
        if len(valid_members) > 0:
            logger.info("  有効なメンバーの例（最初の5名）: %s", list(valid_members[:5]))

        # 保有力量データに含まれるメンバーコードを確認
        acquired_member_codes = acquired_df['メンバーコード'].unique()
        logger.info("\n保有力量データに含まれるメンバー数: %d名", len(acquired_member_codes))
        if len(acquired_member_codes) > 0:
            logger.info("  保有力量データのメンバーの例（最初の5名）: %s", list(acquired_member_codes[:5]))

        # 一致するメンバーを確認
        matching_members = set(valid_members) & set(acquired_member_codes)
        logger.info("\n一致するメンバー数: %d名", len(matching_members))
        if len(matching_members) == 0:
            logger.warning("  ⚠ 有効なメンバーと保有力量データのメンバーが一致しません！")
            logger.warning("  メンバーマスタと保有力量データのメンバーコードの形式を確認してください")

        # 習得力量データ
        member_competence = acquired_df[
            acquired_df['メンバーコード'].isin(valid_members)
        ].copy()

        logger.info("\n有効メンバーでフィルタ後: %d件", len(member_competence))

        if member_competence.empty:
            logger.warning("  ⚠ 有効なメンバーの習得力量データがありません")
            logger.warning("  保有力量データの先頭5件:")
            logger.warning("\n%s", acquired_df.head().to_string())

            # 空のDataFrameでも必要なカラムを持つ構造を返す
            empty_df = pd.DataFrame(columns=[
                'メンバーコード', '力量コード', '力量タイプ', 'レベル',
                '正規化レベル', '力量名', '力量カテゴリー名'
            ])
            return empty_df, valid_members.tolist()

        # レベル正規化
        member_competence['正規化レベル'] = member_competence.apply(
            lambda row: self.normalize_level(row['レベル'], row['力量タイプ']),
            axis=1
        )

        # 力量マスタと結合
        member_competence = member_competence.merge(
            competence_master[['力量コード', '力量名', '力量タイプ', '力量カテゴリー名']],
            left_on='力量コード',
            right_on='力量コード',
            how='left',
            suffixes=('', '_master')
        )

        logger.info("力量マスタと結合後: %d件", len(member_competence))

        return member_competence, valid_members.tolist()
    
    def create_skill_matrix(self, member_competence: pd.DataFrame) -> pd.DataFrame:
        """
        メンバー×力量マトリクスを作成

        Args:
            member_competence: メンバー習得力量データ

        Returns:
            メンバー×力量マトリクス
        """
        logger.info("\n" + "=" * 80)
        logger.info("メンバー×力量マトリクス作成")
        logger.info("=" * 80)

        # 空データの場合は空のマトリクスを返す
        if member_competence.empty:
            logger.warning("  ⚠ メンバー習得力量データが空のため、空のマトリクスを返します")
            return pd.DataFrame()

        # 必須カラムの確認
        required_columns = ['メンバーコード', '力量コード', '正規化レベル']
        missing_columns = [col for col in required_columns if col not in member_competence.columns]
        if missing_columns:
            logger.error("  ⚠ 必須カラムが不足しています: %s", missing_columns)
            logger.error("  利用可能なカラム: %s", list(member_competence.columns))
            raise ValueError(f"メンバー習得力量データに必須カラムが不足しています: {missing_columns}")

        skill_matrix = member_competence.pivot_table(
            index='メンバーコード',
            columns='力量コード',
            values='正規化レベル',
            fill_value=0
        )

        logger.info(
            "\nマトリクスサイズ: %d名 × %d力量",
            skill_matrix.shape[0],
            skill_matrix.shape[1],
        )

        return skill_matrix
    
    def clean_members_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        メンバーマスタをクリーンにする
        
        Args:
            data: 読み込んだデータの辞書
            
        Returns:
            クリーンなメンバーマスタ
        """
        members_clean = data['members'][
            (~data['members']['メンバー名'].str.contains('削除|テスト|test', case=False, na=False)) &
            (data['members']['メンバーコード'].notna())
        ].copy()
        
        # 必要なカラムのみ抽出
        columns = ['メンバーコード', 'メンバー名', 'よみがな', '生年月日', '性別',
                  '入社年月日', '社員区分', '役職', '職能・等級']
        
        # 存在するカラムのみ選択
        available_columns = [col for col in columns if col in members_clean.columns]
        
        return members_clean[available_columns]
