import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

class ArtifactEvaluator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, model_bundle: Dict[str, Any], test_df: pd.DataFrame):
        logger.info("Generating evaluation artifacts...")
        predictor = model_bundle["predictor"]
        
        try:
            importance_df = predictor.get_feature_importance(test_df)
            importance_df.to_csv(self.output_dir / "feature_importance.csv", index=False)
            logger.info("Saved feature_importance.csv")
        except Exception as e:
            logger.warning(f"Could not generate feature importance: {e}")
