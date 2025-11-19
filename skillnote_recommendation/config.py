import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class PathConfig:
    """Project path configuration."""
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    PAGES_DIR: Path = BASE_DIR / "pages"
    
    # Ensure directories exist
    def __post_init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)

@dataclass
class ModelConfig:
    """Default model hyperparameters."""
    RANDOM_STATE: int = 42
    N_JOBS: int = -1
    
    # LiNGAM specific
    LINGAM_PRIOR_KNOWLEDGE_THRESHOLD: float = 0.0
    
    # Graph specific
    GRAPH_WALK_LENGTH: int = 10
    GRAPH_NUM_WALKS: int = 80
    GRAPH_EMBEDDING_DIM: int = 64

@dataclass
class AppConfig:
    """Main application configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    APP_TITLE: str = "CareerNavigator"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("STREAMLIT_DEBUG", "False").lower() == "true"

# Global instance
config = AppConfig()
