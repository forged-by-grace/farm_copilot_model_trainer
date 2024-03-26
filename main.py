from farm_copilot.config.configuration import ConfigurationManager
from farm_copilot.pipeline.stage_01_prepare_base_model import PrepareBaseModelTrainingPipeline
from farm_copilot.utils import logger

STAGE_NAME = "Prepare base model"

try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


