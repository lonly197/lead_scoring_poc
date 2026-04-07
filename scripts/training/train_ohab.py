import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import setup_logging
from src.training.pipeline.data_prep import DataPreparer
from src.training.pipeline.trainer import ModelTrainer
from src.training.pipeline.evaluator import ArtifactEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="OHAB Training Pipeline")
    parser.add_argument("--data-path", type=str, default="./data/final/desensitized.parquet")
    parser.add_argument("--target", type=str, default="线索评级结果")
    parser.add_argument("--preset", type=str, default="medium_quality")
    parser.add_argument("--output-dir", type=str, default="./outputs/models/ohab_model")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(level=logging.INFO)
    
    config_dict = {
        "target": args.target,
        "preset": args.preset,
        "output_dir": args.output_dir
    }
    
    # 1. Prepare Data
    preparer = DataPreparer(config_dict)
    data_bundle = preparer.prepare_data(args.data_path)
    
    # 2. Train Model
    trainer = ModelTrainer(config_dict)
    model_bundle = trainer.fit(data_bundle.train_df, data_bundle.valid_df)
    
    # 3. Evaluate
    evaluator = ArtifactEvaluator(args.output_dir)
    evaluator.generate_all(model_bundle, data_bundle.test_df)

if __name__ == "__main__":
    main()
