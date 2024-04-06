from farm_copilot.config.configuration import ConfigurationManager
from farm_copilot.components.model_trainer import Training
from farm_copilot.utils import logger


STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.load_datasets()
        training.prefetch()
        training.train()



if __name__ == "__main__":
    try:
        logger.info(f"************************")
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started")
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started")
    except Exception as e:
        logger.exception(e)
        raise e
            