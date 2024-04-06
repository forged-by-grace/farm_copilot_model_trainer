import os
from farm_copilot.constants import *
from farm_copilot.utils.common import read_yaml, create_directories
from farm_copilot.entity.config_entity import (PrepareDataConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(self.config.data_ingestion.local_data_file),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_input_shape=self.params.INPUT_SHAPE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_offset=self.params.OFFSET,
        )

        return prepare_base_model_config

    def get_prepare_data_config(self) -> PrepareDataConfig:
        training_data = Path(self.config.data_ingestion.local_data_file)

        prepare_data_config = PrepareDataConfig(training_data=training_data)

        return prepare_data_config


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepared_base_model = self.config.prepare_base_model
        params = self.params
        training_data_path = Path(self.config.data_ingestion.train_dataset)
        val_data_path = Path(self.config.data_ingestion.validation_dataset)

        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepared_base_model.updated_base_model_path),
            training_ds_path=training_data_path,
            validation_ds_path=val_data_path,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_train_split=params.TRAIN_SPLIT,
            params_test_split=params.TEST_SPLIT,
            params_validation_split=params.VALIDATION_SPLIT
        )

        return training_config 


    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation = self.config.evaluation
        test_ds_path = Path(self.config.data_ingestion.test_dataset)

        create_directories([Path(evaluation.root_dir)])

        eval_config = EvaluationConfig(
            path_of_model=Path(evaluation.path_of_model),
            test_ds_path=test_ds_path,
            mlflow_uri=evaluation.mlflow_uri,
            path_to_eval_score=Path(evaluation.path_to_eval_score),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_test_split=self.params.TEST_SPLIT,
            params_train_split=self.params.TRAIN_SPLIT,
            params_validation_split=self.params.VALIDATION_SPLIT
        )
        return eval_config

      