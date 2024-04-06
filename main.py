from farm_copilot.config.configuration import ConfigurationManager
from farm_copilot.pipeline.stage_01_prepare_base_model import PrepareBaseModelPipeline
from farm_copilot.pipeline.stage_04_model_evaluation_pipeline import ModelEvaluationPipeline
from farm_copilot.pipeline.stage_03_model_trainer import ModelTrainingPipeline

from farm_copilot.utils import logger
from dotenv import load_dotenv

load_dotenv()




STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Training"


if __name__ == "__main__":
    try:
        logger.info(f"************************")
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started")
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
          
            
STAGE_NAME = "Model Evaluation"

if __name__ == "__main__":
    try:
        logger.info(f"************************")
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started")
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started")
    except Exception as e:
        logger.exception(e)
        raise e
            


