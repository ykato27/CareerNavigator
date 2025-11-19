import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("Checking imports...")
    from skillnote_recommendation.ml.causal_structure_learner import CausalStructureLearner
    from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender
    from skillnote_recommendation.graph.causal_graph_visualizer import CausalGraphVisualizer
    print("Imports successful.")

    print("Checking instantiation...")
    learner = CausalStructureLearner()
    print("CausalStructureLearner instantiated.")
    
    # Mock data for Recommender
    import pandas as pd
    member_competence = pd.DataFrame({
        'メンバーコード': ['M1', 'M2'], 
        '力量コード': ['S1', 'S2'], 
        '正規化レベル': [1.0, 0.5]
    })
    competence_master = pd.DataFrame({
        '力量コード': ['S1', 'S2'], 
        '力量名': ['Skill1', 'Skill2']
    })
    
    recommender = CausalGraphRecommender(member_competence, competence_master)
    print("CausalGraphRecommender instantiated.")
    
    print("Verification successful!")

except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
