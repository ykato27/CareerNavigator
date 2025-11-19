"""
因果構造学習モジュール

LiNGAM (Linear Non-Gaussian Acyclic Model) を用いて
スキル間の因果関係（構造）をデータから学習します。
計算コスト削減のため、クラスタリングによる分割実行を行います。
"""

import numpy as np
import pandas as pd
import networkx as nx
import lingam
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class CausalStructureLearner:
    """
    因果構造学習クラス
    
    DirectLiNGAMを用いてスキル間の因果構造を学習します。
    大規模なスキルセット（例: 300個以上）に対応するため、
    相関ベースのクラスタリング（コミュニティ検出）を行い、
    分割して因果探索を実行します。
    """
    
    def __init__(
        self, 
        correlation_threshold: float = 0.2,
        min_cluster_size: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            correlation_threshold: クラスタリング用のグラフ構築時の相関閾値
            min_cluster_size: LiNGAMを実行する最小クラスタサイズ
            random_state: 乱数シード
        """
        self.correlation_threshold = correlation_threshold
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        
        # 学習結果
        self.adjacency_matrix_: Optional[pd.DataFrame] = None
        self.causal_effects_: Optional[Dict[str, Dict[str, float]]] = None
        self.clusters_: List[List[str]] = []
        
        self.is_fitted = False

    def fit(self, skill_matrix: pd.DataFrame):
        """
        因果構造を学習
        
        Args:
            skill_matrix: メンバー×スキルのデータフレーム (数値データ)
        """
        logger.info("=" * 60)
        logger.info("因果構造学習 (LiNGAM) 開始")
        logger.info(f"データサイズ: {skill_matrix.shape}")
        logger.info("=" * 60)
        
        # 欠損値処理（念のため）
        if skill_matrix.isnull().any().any():
            logger.warning("欠損値が含まれています。0で埋めます。")
            skill_matrix = skill_matrix.fillna(0)
            
        # 定数列（分散0）の除外
        var = skill_matrix.var()
        valid_cols = var[var > 0].index.tolist()
        if len(valid_cols) < len(skill_matrix.columns):
            logger.info(f"分散0の列を除外: {len(skill_matrix.columns)} -> {len(valid_cols)}")
            skill_matrix = skill_matrix[valid_cols]
            
        self.feature_names = skill_matrix.columns.tolist()
        n_features = len(self.feature_names)
        
        # 結果の隣接行列を初期化（0行列）
        self.adjacency_matrix_ = pd.DataFrame(
            np.zeros((n_features, n_features)),
            index=self.feature_names,
            columns=self.feature_names
        )
        
        # 1. クラスタリング（コミュニティ検出）
        self.clusters_ = self._detect_communities(skill_matrix)
        logger.info(f"コミュニティ分割完了: {len(self.clusters_)}個のクラスタ")
        
        # 2. 分割実行
        total_edges = 0
        for i, cluster_features in enumerate(self.clusters_):
            if len(cluster_features) < self.min_cluster_size:
                logger.info(f"  Cluster {i+1}: スキル数 {len(cluster_features)} (スキップ: サイズ不足)")
                continue
                
            logger.info(f"  Cluster {i+1}: スキル数 {len(cluster_features)} -> LiNGAM実行中...")
            
            try:
                # クラスタ内のデータ抽出
                cluster_data = skill_matrix[cluster_features]
                
                # LiNGAM実行
                model = lingam.DirectLiNGAM(random_state=self.random_state)
                model.fit(cluster_data)
                
                # 隣接行列の取得 (行:原因 -> 列:結果 の形式に変換が必要)
                # lingamのadjacency_matrix_は [to, from] の形式 (B_{ij} is coeff from j to i)
                # つまり 行i, 列j は j -> i の係数
                adj_matrix = model.adjacency_matrix_
                
                # 結果を全体の隣接行列に統合
                # DataFrameにして扱いやすくする
                cluster_adj_df = pd.DataFrame(
                    adj_matrix,
                    index=cluster_features,
                    columns=cluster_features
                )
                
                # 転置して [from, to] 形式にするのが一般的だが、
                # ここでは lingam の定義 (row=to, col=from) を一旦そのまま受け取り、
                # 格納時に [from, to] に変換するか、統一する。
                # ネットワーク分析では通常 adj[i][j] は i -> j を表すことが多いが、
                # SEM/LiNGAMの慣習では x = Bx + e なので B[i][j] は j -> i。
                # ここでは「行=原因(From), 列=結果(To)」の形式（networkx形式）に変換して保存する。
                
                # B[i, j] は j -> i なので、B.T [j, i] が j -> i
                adj_matrix_nx = cluster_adj_df.T
                
                # 全体行列に加算
                self.adjacency_matrix_.loc[cluster_features, cluster_features] = adj_matrix_nx
                
                n_edges = np.count_nonzero(adj_matrix)
                total_edges += n_edges
                logger.info(f"    -> 完了: {n_edges}個の因果関係を発見")
                
            except Exception as e:
                logger.error(f"    -> 失敗 Cluster {i+1}: {e}")
        
        self.is_fitted = True
        logger.info("=" * 60)
        logger.info(f"因果構造学習完了: 合計 {total_edges} 個のエッジを発見")
        logger.info("=" * 60)
        
        return self

    def _detect_communities(self, data: pd.DataFrame) -> List[List[str]]:
        """
        相関行列に基づきコミュニティ検出を行う
        """
        # 相関行列の計算
        corr_matrix = data.corr().abs()
        
        # グラフ構築
        G = nx.Graph()
        features = data.columns.tolist()
        G.add_nodes_from(features)
        
        # 閾値以上の相関を持つエッジを追加
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                u = features[i]
                v = features[j]
                weight = corr_matrix.iloc[i, j]
                if weight > self.correlation_threshold:
                    G.add_edge(u, v, weight=weight)
        
        # コミュニティ検出 (Greedy Modularity)
        # Louvain法が使えればベストだが、標準ライブラリ依存を減らすためnetworkx標準のものを使用
        try:
            communities = nx.community.greedy_modularity_communities(G, weight='weight')
            # frozensetをlistに変換
            return [list(c) for c in communities]
        except Exception as e:
            logger.warning(f"コミュニティ検出に失敗: {e}. 全体を1つのクラスタとして扱います。")
            return [features]

    def get_adjacency_matrix(self) -> pd.DataFrame:
        """
        学習された隣接行列を取得 (行=From, 列=To)
        """
        if not self.is_fitted:
            raise RuntimeError("モデルが学習されていません。fit()を実行してください。")
        return self.adjacency_matrix_

    def get_causal_effects(self) -> Dict[str, Dict[str, float]]:
        """
        総合効果（Total Effects）を計算
        
        Returns:
            {cause_skill: {effect_skill: effect_value}}
        """
        if not self.is_fitted:
            raise RuntimeError("モデルが学習されていません。fit()を実行してください。")
            
        # 隣接行列 (From -> To)
        adj = self.adjacency_matrix_.values
        
        # 総合効果 = (I - B)^-1 - I  (ただしBは To <- From 形式)
        # ここで self.adjacency_matrix_ は From -> To なので、転置して B を作る
        B = adj.T
        
        try:
            # (I - B)^-1
            I = np.eye(len(B))
            total_effects_matrix = np.linalg.inv(I - B) - I
            
            # 再び From -> To 形式に戻すために転置
            total_effects_df = pd.DataFrame(
                total_effects_matrix.T,
                index=self.adjacency_matrix_.index,
                columns=self.adjacency_matrix_.columns
            )
            
            # 辞書形式に変換 (値が小さいものは除外)
            effects = {}
            for col in total_effects_df.columns: # From
                effects[col] = {}
                for row in total_effects_df.index: # To
                    val = total_effects_df.loc[col, row]
                    if abs(val) > 0.01: # 閾値
                        effects[col][row] = val
                        
            return effects
            
        except np.linalg.LinAlgError:
            logger.warning("逆行列の計算に失敗しました（循環がある可能性があります）。直接効果のみを返します。")
            # 直接効果のみ返す
            effects = {}
            for col in self.adjacency_matrix_.columns:
                effects[col] = {}
                for row in self.adjacency_matrix_.index:
                    val = self.adjacency_matrix_.loc[col, row]
                    if abs(val) > 0.01:
                        effects[col][row] = val
            return effects
