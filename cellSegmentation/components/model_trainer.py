import os, sys
import yaml
from cellSegmentation.utils.main_utils import read_yaml_file
from cellSegmentation.logger import logging
from cellSegmentation.exception import AppException
from cellSegmentation.entity.config_entity import ModelTrainerConfig
from cellSegmentation.entity.artifacts_entity import ModelTrainerArtifact
import mlflow
import re
from ultralytics import YOLO


class ModelTrainer:
    def __init__(
            self,
            model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config
        mlflow.set_experiment("ultralytics/yolov8")
        mlflow_location = os.environ['MLFLOW_TRACKING_URI']
        print("********** ", mlflow_location)
        mlflow.set_tracking_uri(mlflow_location)
        mlflow.start_run()

    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            os.system("unzip data.zip")
            os.system("rm data.zip")
            model = YOLO(self.model_trainer_config.weight_name)
            results = model.train(
                batch=8,
                device="cpu",
                data="data.yaml",
                epochs=self.model_trainer_config.no_epochs,
                imgsz=640,
                plots=True
            )
            metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in model.trainer.metrics.items()}
            mlflow.log_metrics(metrics=metrics_dict, step=model.trainer.epoch)
            mlflow.log_artifact('runs/segment/train/confusion_matrix.png')
            mlflow.log_artifact('runs/segment/train/results.csv')
            mlflow.log_artifact("runs/segment/train/weights/best.pt")
            mlflow.log_param("epochs", self.model_trainer_config.no_epochs)
            mlflow.log_param("model", self.model_trainer_config.weight_name)

            # os.system(
            #     f"yolo task=segment mode=train model={self.model_trainer_config.weight_name} "
            #     f"data=data.yaml epochs={self.model_trainer_config.no_epochs} imgsz=640 save=true"
            # )

            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp runs/segment/train/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")

            os.system("rm -rf yolov8s-seg.pt")
            os.system("rm -rf train")
            os.system("rm -rf valid")
            os.system("rm -rf test")
            os.system("rm -rf data.yaml")
            os.system("rm -rf runs")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="artifacts/model_trainer/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            mlflow.end_run()
            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)
