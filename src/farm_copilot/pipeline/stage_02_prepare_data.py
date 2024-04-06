from farm_copilot.config.configuration import ConfigurationManager
from farm_copilot.components.prepare_data import PrepareData
from farm_copilot.utils import logger


STAGE_NAME = "Prepare data"

class PrepareDataPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_data_config = config.get_prepare_data_config()
        prepare_data = PrepareData(config=prepare_data_config)
        prepare_data.clean()


if __name__ == "__main__":
    try:
        logger.info(f"************************")
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started")
        pipeline = PrepareDataPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started")
    except Exception as e:
        logger.exception(e)
        raise e
            