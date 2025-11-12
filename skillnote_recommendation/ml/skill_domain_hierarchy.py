"""
スキル領域階層構造の定義

各スキル領域を初級→中級→上級の3段階に分類し、
SEMでの潜在変数モデル構築に使用します。
"""

from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SkillDomainHierarchy:
    """
    スキル領域の階層構造を定義するクラス

    各領域は3段階のレベルに分かれる:
    - Level 1: 初級（基礎的なスキル）
    - Level 2: 中級（応用的なスキル）
    - Level 3: 上級（専門的なスキル）
    """

    # デフォルトのスキル領域階層定義
    DEFAULT_DOMAIN_HIERARCHY = {
        'プログラミング': {
            'level_1': {
                'keywords': ['基礎', '入門', 'Python基礎', 'Java基礎', 'Git', 'HTML', 'CSS'],
                'description': 'プログラミング初級（基本構文、バージョン管理）',
            },
            'level_2': {
                'keywords': ['Web開発', 'API', 'フレームワーク', 'テスト', 'デバッグ', 'React', 'Vue', 'Django', 'Flask'],
                'description': 'プログラミング中級（Web開発、フレームワーク）',
            },
            'level_3': {
                'keywords': ['システム設計', 'アーキテクチャ', 'パフォーマンス', 'セキュリティ', 'マイクロサービス', 'クラウド'],
                'description': 'プログラミング上級（システム設計、アーキテクチャ）',
            },
        },
        'データベース': {
            'level_1': {
                'keywords': ['SQL基礎', 'データベース基礎', 'SELECT', 'INSERT', 'UPDATE', 'DELETE'],
                'description': 'データベース初級（基本的なSQL操作）',
            },
            'level_2': {
                'keywords': ['JOIN', 'インデックス', 'トランザクション', 'ストアドプロシージャ', 'MySQL', 'PostgreSQL'],
                'description': 'データベース中級（複雑なクエリ、最適化）',
            },
            'level_3': {
                'keywords': ['DB設計', 'パフォーマンスチューニング', 'レプリケーション', 'シャーディング', 'NoSQL', 'Redis'],
                'description': 'データベース上級（設計、運用、NoSQL）',
            },
        },
        'データ分析': {
            'level_1': {
                'keywords': ['Excel', 'データ可視化', '基本統計', 'グラフ作成', 'ピボットテーブル'],
                'description': 'データ分析初級（Excel、基本統計）',
            },
            'level_2': {
                'keywords': ['Python分析', 'pandas', 'numpy', 'matplotlib', 'データクレンジング', 'EDA'],
                'description': 'データ分析中級（Python、探索的データ分析）',
            },
            'level_3': {
                'keywords': ['機械学習', 'モデリング', '統計検定', 'A/Bテスト', 'scikit-learn', 'TensorFlow'],
                'description': 'データ分析上級（機械学習、統計モデリング）',
            },
        },
        'マネジメント': {
            'level_1': {
                'keywords': ['進捗管理', 'タスク管理', 'スケジュール', '報告書作成', '会議運営'],
                'description': 'マネジメント初級（進捗管理、タスク管理）',
            },
            'level_2': {
                'keywords': ['プロジェクト管理', 'リーダーシップ', 'チームビルディング', 'リスク管理', '目標設定'],
                'description': 'マネジメント中級（プロジェクト管理、チーム運営）',
            },
            'level_3': {
                'keywords': ['戦略立案', '予算管理', '事業計画', '組織設計', '経営判断'],
                'description': 'マネジメント上級（戦略、事業計画）',
            },
        },
        'コミュニケーション': {
            'level_1': {
                'keywords': ['報連相', 'メール', 'ビジネスマナー', '電話応対', '来客対応'],
                'description': 'コミュニケーション初級（報連相、ビジネスマナー）',
            },
            'level_2': {
                'keywords': ['プレゼンテーション', '提案書作成', 'ファシリテーション', '交渉', 'ヒアリング'],
                'description': 'コミュニケーション中級（プレゼン、提案）',
            },
            'level_3': {
                'keywords': ['経営層コミュニケーション', 'ステークホルダー管理', '危機管理広報', '組織変革'],
                'description': 'コミュニケーション上級（経営層、ステークホルダー管理）',
            },
        },
    }

    def __init__(
        self,
        competence_master: pd.DataFrame,
        custom_hierarchy: Optional[Dict] = None,
    ):
        """
        Args:
            competence_master: 力量マスタ（力量名、力量カテゴリー名を含む）
            custom_hierarchy: カスタム階層定義（Noneの場合はデフォルトを使用）
        """
        self.competence_master = competence_master
        self.hierarchy = custom_hierarchy or self.DEFAULT_DOMAIN_HIERARCHY

        # 力量をドメイン・レベルに分類
        self.competence_classification = self._classify_competences()

        logger.info("\nSkill Domain Hierarchy 構築完了")
        logger.info("  領域数: %d", len(self.hierarchy))
        logger.info("  分類された力量数: %d", len(self.competence_classification))

    def _classify_competences(self) -> Dict[str, Dict[str, any]]:
        """
        各力量をドメイン・レベルに分類

        Returns:
            {力量コード: {'domain': str, 'level': int, 'competence_name': str}}
        """
        classification = {}

        for _, row in self.competence_master.iterrows():
            competence_code = row['力量コード']
            competence_name = row['力量名']

            # 力量カテゴリー名がある場合はそれを優先的に使用
            competence_category = row.get('力量カテゴリー名', None)

            # カテゴリーベースでドメインとレベルを判定
            domain, level = self._match_domain_level_with_category(
                competence_name, competence_category
            )

            if domain is not None:
                classification[competence_code] = {
                    'domain': domain,
                    'level': level,
                    'competence_name': competence_name,
                }

        return classification

    def _match_domain_level_with_category(
        self, competence_name: str, competence_category: Optional[str]
    ) -> tuple:
        """
        力量カテゴリーと力量名からドメインとレベルを判定

        Args:
            competence_name: 力量名
            competence_category: 力量カテゴリー名（存在する場合）

        Returns:
            (domain, level) または (None, None)
        """
        # カテゴリーが存在する場合、カテゴリーベースでドメインを判定
        if competence_category and not pd.isna(competence_category):
            domain = self._map_category_to_domain(competence_category)
            if domain is not None:
                # レベルは力量名から判定
                level = self._infer_level_from_name(competence_name)
                return (domain, level)

        # カテゴリーがない場合、従来のキーワードマッチング
        return self._match_domain_level(competence_name)

    def _map_category_to_domain(self, category: str) -> Optional[str]:
        """
        力量カテゴリー名をドメインにマッピング

        Args:
            category: 力量カテゴリー名

        Returns:
            ドメイン名（該当なしの場合はNone）
        """
        category_lower = category.lower()

        # カテゴリー名とドメインのマッピング
        category_mapping = {
            'プログラミング': ['プログラミング', 'コーディング', '開発', 'python', 'java', 'javascript', 'web', 'アプリ'],
            'データベース': ['データベース', 'db', 'sql', 'mysql', 'postgresql', 'oracle', 'nosql'],
            'データ分析': ['データ分析', '分析', 'データサイエンス', '統計', '機械学習', 'ai', 'bi', 'tableau'],
            'マネジメント': ['マネジメント', '管理', 'プロジェクト', 'リーダー', '組織', '戦略', '計画'],
            'コミュニケーション': ['コミュニケーション', 'プレゼン', '交渉', '報告', '文書作成', '提案'],
        }

        for domain, keywords in category_mapping.items():
            if any(keyword in category_lower for keyword in keywords):
                return domain

        return None

    def _infer_level_from_name(self, competence_name: str) -> int:
        """
        力量名からレベルを推測

        Args:
            competence_name: 力量名

        Returns:
            レベル（1～3）
        """
        competence_name_lower = competence_name.lower()

        # レベル1: 基礎・入門・初級
        if any(keyword in competence_name_lower for keyword in ['基礎', '基本', '入門', '初級', '初歩']):
            return 1

        # レベル3: 上級・専門・設計
        if any(keyword in competence_name_lower for keyword in ['上級', '専門', '設計', 'アーキテクチャ', '戦略', '統括']):
            return 3

        # レベル2: 応用・実践・中級（デフォルト）
        return 2

    def _match_domain_level(self, competence_name: str) -> tuple:
        """
        力量名からドメインとレベルを判定（キーワードマッチング）

        Args:
            competence_name: 力量名

        Returns:
            (domain, level) または (None, None)
        """
        competence_name_lower = competence_name.lower()

        best_match = None
        best_match_score = 0

        for domain, levels in self.hierarchy.items():
            for level_key, level_info in levels.items():
                keywords = level_info['keywords']

                # キーワードマッチング
                match_count = sum(
                    1 for keyword in keywords
                    if keyword.lower() in competence_name_lower
                )

                if match_count > best_match_score:
                    best_match_score = match_count
                    level_num = int(level_key.split('_')[1])
                    best_match = (domain, level_num)

        if best_match_score > 0:
            return best_match

        # マッチしない場合、カテゴリー名から推測
        return self._fallback_classification(competence_name)

    def _fallback_classification(self, competence_name: str) -> tuple:
        """
        キーワードマッチングで判定できない場合のフォールバック

        カテゴリー名から推測
        """
        # 「基礎」「基本」「入門」→ Level 1
        if any(keyword in competence_name for keyword in ['基礎', '基本', '入門', '初級']):
            # ドメインは最も一般的な「プログラミング」にフォールバック
            return ('プログラミング', 1)

        # 「応用」「実践」「開発」→ Level 2
        if any(keyword in competence_name for keyword in ['応用', '実践', '開発', '中級']):
            return ('プログラミング', 2)

        # 「設計」「戦略」「上級」→ Level 3
        if any(keyword in competence_name for keyword in ['設計', '戦略', '上級', 'アーキテクチャ']):
            return ('プログラミング', 3)

        # それ以外はNone
        return (None, None)

    def get_domain(self, competence_code: str) -> Optional[str]:
        """力量のドメインを取得"""
        info = self.competence_classification.get(competence_code)
        return info['domain'] if info else None

    def get_level(self, competence_code: str) -> Optional[int]:
        """力量のレベルを取得（1=初級, 2=中級, 3=上級）"""
        info = self.competence_classification.get(competence_code)
        return info['level'] if info else None

    def get_competences_by_domain(self, domain: str) -> List[str]:
        """特定のドメインの力量コードリストを取得"""
        return [
            code for code, info in self.competence_classification.items()
            if info['domain'] == domain
        ]

    def get_competences_by_level(self, domain: str, level: int) -> List[str]:
        """特定のドメイン・レベルの力量コードリストを取得"""
        return [
            code for code, info in self.competence_classification.items()
            if info['domain'] == domain and info['level'] == level
        ]

    def get_next_level_competences(self, competence_code: str) -> List[str]:
        """
        指定した力量の次のレベルの力量を取得

        Args:
            competence_code: 基準となる力量コード

        Returns:
            次のレベルの力量コードリスト
        """
        info = self.competence_classification.get(competence_code)
        if info is None or info['level'] >= 3:
            return []

        domain = info['domain']
        next_level = info['level'] + 1

        return self.get_competences_by_level(domain, next_level)

    def get_domain_progression(self, domain: str) -> Dict[int, List[str]]:
        """
        特定のドメインの進行経路を取得

        Returns:
            {level: [力量コードリスト]}
        """
        progression = {1: [], 2: [], 3: []}

        for code, info in self.competence_classification.items():
            if info['domain'] == domain:
                progression[info['level']].append(code)

        return progression

    def get_domain_statistics(self) -> pd.DataFrame:
        """
        各ドメイン・レベルの力量数統計を取得

        Returns:
            DataFrame with columns: [Domain, Level_1, Level_2, Level_3, Total]
        """
        stats = []

        for domain in self.hierarchy.keys():
            level_1_count = len(self.get_competences_by_level(domain, 1))
            level_2_count = len(self.get_competences_by_level(domain, 2))
            level_3_count = len(self.get_competences_by_level(domain, 3))

            stats.append({
                'Domain': domain,
                'Level_1': level_1_count,
                'Level_2': level_2_count,
                'Level_3': level_3_count,
                'Total': level_1_count + level_2_count + level_3_count,
            })

        return pd.DataFrame(stats)
