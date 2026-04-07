from pathlib import Path
import logging
from dataclasses import dataclass
import pandas as pd
from typing import Dict, Any

from src.data.loader import DataLoader, FeatureEngineer, smart_split_data
from src.data.feature_screening import clean_raw_schema

logger = logging.getLogger(__name__)

@dataclass
class DataBundle:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_metadata: Dict[str, Any]

class DataPreparer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def prepare_data(self, data_path: str) -> DataBundle:
        logger.info("Loading data...")
        loader = DataLoader(data_path, auto_adapt=True)
        df = loader.load()
        target_label = self.config["target"]
        
        logger.info("Splitting data...")
        train_df, valid_df, test_df, split_info = smart_split_data(
            df, target_label, split_mode=self.config.get("split_mode", "random")
        )
        
        logger.info("Feature engineering...")
        train_df, raw_schema_report = clean_raw_schema(train_df, target_label=target_label)
        feature_engineer = FeatureEngineer(
            time_columns=[], numeric_columns=[] # Adjust based on config
        )
        train_df, feature_metadata = feature_engineer.fit_transform(train_df)
        valid_df, _ = feature_engineer.transform(valid_df, interaction_context=feature_metadata.get("interaction_context"))
        test_df, _ = feature_engineer.transform(test_df, interaction_context=feature_metadata.get("interaction_context"))
        
        feature_metadata["split_info"] = split_info
        
        return DataBundle(train_df, valid_df, test_df, feature_metadata)
