"""
データローダー

CSVファイルからデータを読み込む機能を提供
"""

import pandas as pd
import re
import os
from typing import Dict
from config import Config


class DataLoader:
    """データ読み込みクラス"""
    
    def __init__(self, data_dir: str = None):
        """
        初期化
        
        Args:
            data_dir: データディレクトリのパス（Noneの場合はConfig.DATA_DIRを使用）
        """
        self.data_dir = data_dir or Config.DATA_DIR
    
    @staticmethod
    def clean_column_name(col_name: str) -> str:
        """
        カラム名をクリーンにする
        
        Args:
            col_name: 元のカラム名
            
        Returns:
            クリーンなカラム名
        """
        col_name = re.sub(r'\s*###\[.*?\]###', '', col_name)
        return col_name.strip()
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        CSVファイルを読み込む
        
        Args:
            filename: ファイル名
            
        Returns:
            DataFrame
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} が見つかりません")
        
        df = pd.read_csv(filepath, encoding=Config.FILE_ENCODING)
        df.columns = [self.clean_column_name(col) for col in df.columns]
        
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        全てのCSVファイルを読み込む
        
        Returns:
            データの辞書（キー: ファイル種別、値: DataFrame）
        """
        print("=" * 80)
        print("データ読み込み")
        print("=" * 80)
        
        data = {}
        
        for key, filename in Config.INPUT_FILES.items():
            try:
                df = self.load_csv(filename)
                data[key] = df
                print(f"  ✓ {filename}: {len(df)}行")
            except FileNotFoundError as e:
                print(f"  ✗ {filename}: ファイルが見つかりません")
                raise e
        
        print(f"\n全{len(data)}ファイルの読み込み完了")
        
        return data
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        データの整合性をチェック
        
        Args:
            data: データの辞書
            
        Returns:
            検証結果（True: 正常、False: 異常）
        """
        print("\n" + "=" * 80)
        print("データ検証")
        print("=" * 80)
        
        # 必須カラムのチェック
        required_columns = {
            'members': ['メンバーコード', 'メンバー名'],
            'acquired': ['メンバーコード', '力量コード', '力量タイプ', 'レベル'],
            'skills': ['力量コード', '力量名'],
            'education': ['力量コード', '力量名'],
            'license': ['力量コード', '力量名'],
            'categories': ['力量カテゴリーコード']
        }
        
        all_valid = True
        
        for key, required_cols in required_columns.items():
            df = data[key]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  ✗ {key}: 必須カラムが不足 - {missing_cols}")
                all_valid = False
            else:
                print(f"  ✓ {key}: 必須カラム確認")
        
        if all_valid:
            print("\n全データの検証完了")
        else:
            print("\n検証エラーがあります")
        
        return all_valid
