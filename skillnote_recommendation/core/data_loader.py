"""
データローダー

CSVファイルからデータを読み込む機能を提供
ディレクトリスキャンにより複数CSVファイルの自動読み込みと結合に対応
"""

import logging
import pandas as pd
import re
import os
import glob
from typing import Dict
from skillnote_recommendation.core.config import Config


logger = logging.getLogger(__name__)


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

    def load_csv_from_directory(self, dir_name: str) -> pd.DataFrame:
        """
        ディレクトリ内の全CSVファイルを読み込んで結合

        Args:
            dir_name: ディレクトリ名

        Returns:
            結合されたDataFrame

        Raises:
            FileNotFoundError: ディレクトリが存在しない、またはCSVファイルがない場合
            ValueError: カラム構造が異なるファイルがある場合
        """
        dir_path = os.path.join(self.data_dir, dir_name)

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"ディレクトリ {dir_path} が見つかりません")

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"{dir_path} はディレクトリではありません")

        # ディレクトリ内の全CSVファイルを検索
        csv_files = glob.glob(os.path.join(dir_path, '*.csv'))

        if not csv_files:
            raise FileNotFoundError(f"ディレクトリ {dir_path} にCSVファイルが見つかりません")

        # 全CSVファイルを読み込んで結合
        dfs = []
        first_columns = None

        for csv_file in sorted(csv_files):  # ソートして順序を一定に
            df = pd.read_csv(csv_file, encoding=Config.FILE_ENCODING)
            df.columns = [self.clean_column_name(col) for col in df.columns]

            # カラム構造の整合性チェック
            if first_columns is None:
                first_columns = set(df.columns)
                first_file = os.path.basename(csv_file)
            else:
                current_columns = set(df.columns)
                if first_columns != current_columns:
                    missing_in_current = first_columns - current_columns
                    extra_in_current = current_columns - first_columns

                    error_msg = (
                        f"カラム構造が一致しません:\n"
                        f"  基準ファイル: {first_file}\n"
                        f"  問題ファイル: {os.path.basename(csv_file)}\n"
                    )
                    if missing_in_current:
                        error_msg += f"  不足カラム: {sorted(missing_in_current)}\n"
                    if extra_in_current:
                        error_msg += f"  余分なカラム: {sorted(extra_in_current)}\n"

                    raise ValueError(error_msg)

            dfs.append(df)

        # 全てのDataFrameを結合
        combined_df = pd.concat(dfs, ignore_index=True)

        return combined_df

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        全てのディレクトリからCSVファイルを読み込む

        Returns:
            データの辞書（キー: ファイル種別、値: DataFrame）
        """
        logger.info("=" * 80)
        logger.info("データ読み込み")
        logger.info("=" * 80)

        data = {}

        for key, dir_name in Config.INPUT_DIRS.items():
            try:
                df = self.load_csv_from_directory(dir_name)
                data[key] = df

                # ファイル数も表示
                dir_path = os.path.join(self.data_dir, dir_name)
                csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
                logger.info(
                    "  ✓ %s/: %dファイル, %d行",
                    dir_name,
                    len(csv_files),
                    len(df),
                )

            except FileNotFoundError as e:
                logger.error("  ✗ %s: ディレクトリ '%s/' が見つかりません", key, dir_name)
                raise e

        logger.info("\n全%d種類のデータ読み込み完了", len(data))

        return data
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        データの整合性をチェック
        
        Args:
            data: データの辞書
            
        Returns:
            検証結果（True: 正常、False: 異常）
        """
        logger.info("\n" + "=" * 80)
        logger.info("データ検証")
        logger.info("=" * 80)
        
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
                logger.error("  ✗ %s: 必須カラムが不足 - %s", key, missing_cols)
                all_valid = False
            else:
                logger.info("  ✓ %s: 必須カラム確認", key)

        if all_valid:
            logger.info("\n全データの検証完了")
        else:
            logger.error("\n検証エラーがあります")
        
        return all_valid
