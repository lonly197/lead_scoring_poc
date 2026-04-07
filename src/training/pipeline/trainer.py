import logging
from typing import Dict, Any
import pandas as pd
from src.models.predictor import LeadScoringPredictor

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Training single stage model...")
        output_dir = self.config.get("output_dir", "./outputs/models/ohab_model")
        
        predictor = LeadScoringPredictor(
            label=self.config["target"],
            output_path=output_dir,
            eval_metric=self.config.get("eval_metric", "balanced_accuracy"),
            problem_type="multiclass"
        )
        
        predictor.train(
            train_data=train_df,
            tuning_data=valid_df if len(valid_df) > 0 else None,
            presets=self.config.get("preset", "medium_quality"),
            time_limit=self.config.get("time_limit", 3600)
        )
        
        return {"predictor": predictor, "best_model": predictor.get_model_info().get("best_model")}
