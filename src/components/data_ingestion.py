import os
import os.path
import sys
import logging
from src.exception import CustomException
from src.logger import logging as logger  # Assuming this returns a logger instance
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.raw_data_path = None  # Store raw data path for return

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion method/component")
        try:
            # Read the dataset
            df = pd.read_csv('notebook/stud (1).csv')  # Fixed path separator
            logger.info("Dataset read as DataFrame successfully")
            
            # Create artifacts directory and save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            self.raw_data_path = self.ingestion_config.raw_data_path
            logger.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")

            # Train-test split (Fixed: was logging string instead of actual call)
            logger.info("Performing train-test split (test_size=0.2, random_state=42)")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save train and test sets (Fixed: test_set was getting train_set)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  # Fixed
            
            logger.info(f"Data ingestion completed successfully. Train: {len(train_set)}, Test: {len(test_set)}")
            
            # Return paths
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path  # Added raw data path
            )
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}", exc_info=True)
            raise CustomException(e, sys) from e  # Fixed: proper exception chaining


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path, raw_path = obj.initiate_data_ingestion()  # Fixed return handling
    print(f"Data ingestion complete. Paths: train={train_path}, test={test_path}, raw={raw_path}")