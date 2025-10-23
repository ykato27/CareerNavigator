"""
類似度計算

力量間の類似度（同時習得パターン）を計算
"""

import pandas as pd
import numpy as np
from config import Config


class SimilarityCalculator:
    """類似度計算クラス"""
    
    def __init__(self, sample_size: int = None, threshold: float = None):
        """
        初期化
        
        Args:
            sample_size: サンプリングサイズ（Noneの場合はConfigから取得）
            threshold: 類似度の閾値（Noneの場合はConfigから取得）
        """
        self.sample_size = sample_size or Config.RECOMMENDATION_PARAMS['similarity_sample_size']
        self.threshold = threshold or Config.RECOMMENDATION_PARAMS['similarity_threshold']
    
    def calculate_similarity(self, member_competence: pd.DataFrame) -> pd.DataFrame:
        """
        力量間の類似度を計算（Jaccard係数）
        
        Args:
            member_competence: 会員習得力量データ
            
        Returns:
            類似度データフレーム
        """
        print("\n" + "=" * 80)
        print("力量間類似度計算")
        print("=" * 80)
        
        # 二値マトリクス作成
        skill_matrix = member_competence.pivot_table(
            index='メンバーコード',
            columns='力量コード',
            values='正規化レベル',
            fill_value=0
        )
        skill_binary = (skill_matrix > 0).astype(int)
        
        competences = skill_binary.columns.tolist()
        similarities = []
        
        # サンプリング
        actual_sample_size = min(self.sample_size, len(competences))
        sample_competences = np.random.choice(competences, actual_sample_size, replace=False)
        
        print(f"\n{actual_sample_size}個の力量についてサンプリング計算中...")
        
        for i, comp1 in enumerate(sample_competences):
            if i % 20 == 0 and i > 0:
                print(f"  進捗: {i}/{actual_sample_size}")
            
            comp1_vector = skill_binary[comp1].values
            
            for comp2 in competences:
                if comp1 >= comp2:
                    continue
                
                comp2_vector = skill_binary[comp2].values
                
                # 両方とも習得者がいない組み合わせはスキップ
                if comp1_vector.sum() == 0 or comp2_vector.sum() == 0:
                    continue
                
                # Jaccard類似度を計算
                intersection = (comp1_vector & comp2_vector).sum()
                union = (comp1_vector | comp2_vector).sum()
                
                if union > 0:
                    similarity = intersection / union
                    
                    # 閾値以上の類似度のみ保存
                    if similarity > self.threshold:
                        similarities.append({
                            '力量1': comp1,
                            '力量2': comp2,
                            '類似度': similarity
                        })
        
        similarity_df = pd.DataFrame(similarities)
        print(f"完了: {len(similarity_df)}件の類似ペアを検出")
        
        return similarity_df
