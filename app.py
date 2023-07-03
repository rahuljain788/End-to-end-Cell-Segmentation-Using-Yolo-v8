from cellSegmentation.logger import logging
from cellSegmentation.exception import AppException
import sys

from cellSegmentation.pipeline.training_pipeline import TrainingPipeline

obj = TrainingPipeline()
obj.run_pipeline()
print("Training Done")

