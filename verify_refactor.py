import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("Testing imports...")
    try:
        from skillnote_recommendation.config import config
        print(f"Config loaded. DEBUG={config.DEBUG}")
        
        from skillnote_recommendation.utils.logger import setup_logger
        logger = setup_logger("test_logger")
        logger.info("Logger test")
        
        from skillnote_recommendation.ml.causal_structure_learner import CausalStructureLearner
        learner = CausalStructureLearner()
        print("CausalStructureLearner instantiated.")
        
        from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender
        # Mock dataframes for init
        import pandas as pd
        df_mock = pd.DataFrame()
        recommender = CausalGraphRecommender(df_mock, df_mock)
        print("CausalGraphRecommender instantiated.")
        
        from skillnote_recommendation.ml.unified_sem_estimator import UnifiedSEMEstimator, MeasurementModelSpec, StructuralModelSpec
        ms = [MeasurementModelSpec("L1", ["O1"])]
        ss = [StructuralModelSpec("L1", "L1")] # Invalid but checks init
        sem = UnifiedSEMEstimator(ms, ss)
        print("UnifiedSEMEstimator instantiated.")
        
        print("All imports and instantiations successful.")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_imports():
        sys.exit(0)
    else:
        sys.exit(1)
