import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
import structlog

logger = structlog.get_logger()

try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    # ダミー型定義（型ヒント用）
    BayesianNetwork = Any
    VariableElimination = Any
    logger.warning("pgmpy not found. Bayesian Network features will be disabled.")

class BayesianNetworkRecommender:
    """
    ベイジアンネットワークを用いたスキル推薦クラス
    LiNGAMで学習した因果構造（DAG）を利用し、確率的な推論を行う。
    """
    
    def __init__(self, adj_matrix: pd.DataFrame):
        """
        Args:
            adj_matrix: LiNGAMなどで学習した隣接行列（行:原因 -> 列:結果）
                        値は因果係数。0以外をエッジとみなす。
        """
        self.adj_matrix = adj_matrix
        self.model: Optional[BayesianNetwork] = None
        self.inference: Optional[VariableElimination] = None
        self.nodes: List[str] = []
        
    def fit(self, data: pd.DataFrame, threshold: float = 0.01):
        """
        ベイジアンネットワークのパラメータを学習する
        
        Args:
            data: メンバーごとのスキル保有データ（0/1）
            threshold: エッジを採用する係数の閾値（絶対値）
        """
        if not PGMPY_AVAILABLE:
            logger.warning("pgmpy is not available. Skipping fit.")
            return

        logger.info("Starting Bayesian Network training")
        
        # 1. 隣接行列からエッジリストを作成（DAGの構築）
        edges = []
        self.nodes = list(self.adj_matrix.index)
        
        # データに含まれるカラムのみを対象にする（整合性確保）
        valid_nodes = [col for col in self.nodes if col in data.columns]
        
        for source in valid_nodes:
            for target in valid_nodes:
                if source == target:
                    continue
                
                weight = self.adj_matrix.loc[source, target]
                if abs(weight) >= threshold:
                    edges.append((source, target))
        
        if not edges:
            logger.warning("No edges found with current threshold. Bayesian Network cannot be built effectively.")
            # エッジがない場合でも、独立したノードとしてモデル化は可能
            self.model = BayesianNetwork()
            self.model.add_nodes_from(valid_nodes)
        else:
            # 閉路のチェックと除去（LiNGAMはDAGを保証するはずだが、念のため）
            try:
                # networkxで閉路チェック
                G = nx.DiGraph(edges)
                if not nx.is_directed_acyclic_graph(G):
                    logger.warning("Cycle detected in graph. Removing cycles for Bayesian Network.")
                    # 簡易的な閉路除去: 最小帰還辺集合問題はNP困難なので、単純に閉路を構成するエッジを削除
                    cycles = list(nx.simple_cycles(G))
                    for cycle in cycles:
                        # 閉路の最後のエッジを削除（簡易対応）
                        if (cycle[-1], cycle[0]) in edges:
                            edges.remove((cycle[-1], cycle[0]))
                            
                self.model = BayesianNetwork(edges)
                # データに存在しないノードがエッジに含まれている場合の対処
                # (valid_nodesでフィルタリングしているので基本的には起きないはず)
                
            except Exception as e:
                logger.error(f"Error constructing DAG: {e}")
                raise e

        # 2. パラメータ学習（CPTの計算）
        # データは 0/1 のバイナリであることを想定
        try:
            # 状態名（0, 1）を明示的に指定しないと、データに含まれない状態が無視される可能性がある
            # state_names = {col: [0, 1] for col in valid_nodes} 
            # pgmpyのfitメソッドはstate_namesを受け取らない場合、データから推測する
            
            self.model.fit(
                data[valid_nodes], 
                estimator=MaximumLikelihoodEstimator
            )
            
            # 3. 推論エンジンの初期化
            self.inference = VariableElimination(self.model)
            logger.info("Bayesian Network training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during parameter learning: {e}")
            raise e

    def predict_probability(
        self, 
        member_skills: List[str], 
        target_skill: str
    ) -> float:
        """
        特定のターゲットスキルを保有している確率を推論する
        
        Args:
            member_skills: メンバーが保有しているスキル名のリスト（エビデンス）
            target_skill: 確率を知りたいスキル名
            
        Returns:
            float: 確率 P(Target=1 | Evidence)
        """
        if not PGMPY_AVAILABLE:
            return 0.0

        if self.inference is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
            
        if target_skill not in self.model.nodes():
            logger.warning(f"Target skill '{target_skill}' not in model nodes.")
            return 0.0
            
        # エビデンスの構築 ({'SkillA': 1, 'SkillB': 1, ...})
        # モデルに含まれるノードのみをエビデンスとする
        evidence = {
            skill: 1 
            for skill in member_skills 
            if skill in self.model.nodes() and skill != target_skill
        }
        
        try:
            # 推論実行
            # P(target_skill | evidence)
            result = self.inference.query(
                variables=[target_skill], 
                evidence=evidence,
                show_progress=False
            )
            
            # target_skill = 1 の確率を取得
            # result.values は [P(0), P(1)] の順（通常）
            # state_namesを確認して安全に取得するのがベストだが、
            # バイナリデータ(0, 1)で学習していれば、インデックス1が確率1に対応するはず
            
            # pgmpyのバージョンによって挙動が違う場合があるため、安全策
            prob = result.get_value(**{target_skill: 1})
            
            return float(prob)
            
        except Exception as e:
            logger.error(f"Error during inference for {target_skill}: {e}")
            return 0.0

    def recommend(
        self, 
        member_skills: List[str], 
        candidate_skills: List[str], 
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        メンバーに対して、習得確率が高いスキルを推薦する
        （「このスキル構成なら、次はこのスキルを習得している可能性が高い（自然な流れである）」という推薦）
        
        Args:
            member_skills: 保有スキルリスト
            candidate_skills: 推薦候補スキルリスト（未保有スキル）
            top_n: 返却数
            
        Returns:
            List[Dict]: 推薦結果のリスト
        """
        results = []
        
        for skill in candidate_skills:
            prob = self.predict_probability(member_skills, skill)
            if prob > 0:
                results.append({
                    'skill_name': skill,
                    'probability': prob
                })
                
        # 確率の降順でソート
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return results[:top_n]
