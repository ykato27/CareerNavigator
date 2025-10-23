"""
データ変換

スキルノートのデータを推薦システム用に変換
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from skillnote_recommendation.core.config import Config


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
        print("\n" + "=" * 80)
        print("統合力量マスタ作成")
        print("=" * 80)
        
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
        
        print(f"\n統合完了: {len(competence_master)}件")
        print(f"  - SKILL: {len(competence_master[competence_master['力量タイプ'] == 'SKILL'])}件")
        print(f"  - EDUCATION: {len(competence_master[competence_master['力量タイプ'] == 'EDUCATION'])}件")
        print(f"  - LICENSE: {len(competence_master[competence_master['力量タイプ'] == 'LICENSE'])}件")
        
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
        会員習得力量データを作成
        
        Args:
            data: 読み込んだデータの辞書
            competence_master: 統合力量マスタ
            
        Returns:
            (会員習得力量データ, 有効な会員コードリスト)
        """
        print("\n" + "=" * 80)
        print("会員習得力量データ作成")
        print("=" * 80)
        
        # 有効な会員を抽出
        valid_members = data['members'][
            (~data['members']['メンバー名'].str.contains('削除|テスト|test', case=False, na=False)) &
            (data['members']['メンバーコード'].notna())
        ]['メンバーコード'].unique()
        
        print(f"\n有効な会員数: {len(valid_members)}名")
        
        # 習得力量データ
        member_competence = data['acquired'][
            data['acquired']['メンバーコード'].isin(valid_members)
        ].copy()
        
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
        
        print(f"習得記録数: {len(member_competence)}件")
        
        return member_competence, valid_members.tolist()
    
    def create_skill_matrix(self, member_competence: pd.DataFrame) -> pd.DataFrame:
        """
        会員×力量マトリクスを作成
        
        Args:
            member_competence: 会員習得力量データ
            
        Returns:
            会員×力量マトリクス
        """
        print("\n" + "=" * 80)
        print("会員×力量マトリクス作成")
        print("=" * 80)
        
        skill_matrix = member_competence.pivot_table(
            index='メンバーコード',
            columns='力量コード',
            values='正規化レベル',
            fill_value=0
        )
        
        print(f"\nマトリクスサイズ: {skill_matrix.shape[0]}名 × {skill_matrix.shape[1]}力量")
        
        return skill_matrix
    
    def clean_members_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        会員マスタをクリーンにする
        
        Args:
            data: 読み込んだデータの辞書
            
        Returns:
            クリーンな会員マスタ
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
