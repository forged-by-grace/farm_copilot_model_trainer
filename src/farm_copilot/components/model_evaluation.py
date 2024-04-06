from tensorflow import keras
from farm_copilot.entity.config_entity import EvaluationConfig
from pathlib import Path
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.farm_copilot.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config


    def get_trained_model(self) -> keras.Model:
        self.model = keras.models.load_model(filepath=self.config.path_of_model)


    def evaluation(self):
        self.get_trained_model()
        self.load_test_dataset()
        self.score = self.model.evaluate(self.test_ds)
        self.save_score()
    

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path(self.config.path_to_eval_score), data=scores)

    
    def load_test_dataset(self):
        self.test_ds = keras.preprocessing.image_dataset_from_directory(
                        directory=self.config.test_ds_path,
                        shuffle=True,
                        label_mode='categorical',
                        image_size = tuple(self.config.params_image_size), 
                        batch_size = self.config.params_batch_size,
                        )
        

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
               mlflow.keras.log_model(self.model, "model", registered_model_name="farm_copilot_model")
            else:
                mlflow.keras.log_model(self.model, "model")
