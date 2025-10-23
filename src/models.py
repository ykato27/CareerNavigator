"""
データモデル

スキルノートのデータ構造を定義
"""

import pandas as pd
from typing import Optional


class Member:
    """会員クラス"""
    
    def __init__(self, member_code: str, name: str, role: str = None, grade: str = None):
        self.member_code = member_code
        self.name = name
        self.role = role
        self.grade = grade
    
    def __repr__(self):
        return f"Member(code={self.member_code}, name={self.name})"


class Competence:
    """力量クラス"""
    
    def __init__(self, competence_code: str, name: str, competence_type: str,
                 category: str = None, description: str = None):
        self.competence_code = competence_code
        self.name = name
        self.competence_type = competence_type  # SKILL, EDUCATION, LICENSE
        self.category = category
        self.description = description
    
    def __repr__(self):
        return f"Competence(code={self.competence_code}, name={self.name}, type={self.competence_type})"


class MemberCompetence:
    """会員習得力量クラス"""
    
    def __init__(self, member_code: str, competence_code: str, 
                 level: int, acquired_date: Optional[str] = None):
        self.member_code = member_code
        self.competence_code = competence_code
        self.level = level  # 正規化済みレベル（0-5）
        self.acquired_date = acquired_date
    
    def __repr__(self):
        return f"MemberCompetence(member={self.member_code}, competence={self.competence_code}, level={self.level})"


class Recommendation:
    """推薦結果クラス"""
    
    def __init__(self, competence_code: str, competence_name: str, 
                 competence_type: str, category: str,
                 priority_score: float, category_importance: float,
                 acquisition_ease: float, popularity: float, reason: str):
        self.competence_code = competence_code
        self.competence_name = competence_name
        self.competence_type = competence_type
        self.category = category
        self.priority_score = priority_score
        self.category_importance = category_importance
        self.acquisition_ease = acquisition_ease
        self.popularity = popularity
        self.reason = reason
    
    def to_dict(self):
        """辞書形式に変換"""
        return {
            '力量コード': self.competence_code,
            '力量名': self.competence_name,
            '力量タイプ': self.competence_type,
            'カテゴリ': self.category,
            '優先度スコア': round(self.priority_score, 2),
            'カテゴリ重要度': round(self.category_importance, 2),
            '習得容易性': round(self.acquisition_ease, 2),
            '人気度': round(self.popularity, 2),
            '推薦理由': self.reason
        }
    
    def __repr__(self):
        return f"Recommendation(competence={self.competence_name}, score={self.priority_score:.2f})"
