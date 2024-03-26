from farm_copilot.config.configuration import ConfigurationManager
from farm_copilot.components.prepare_base_model import PrepareBaseModel
from farm_copilot.utils import logger


STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepared_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == "__main__":
    try:
        logger.info(f"************************")
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started")
        pipeline = PrepareBaseModelTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started")
    except Exception as e:
        logger.exception(e)
        raise e