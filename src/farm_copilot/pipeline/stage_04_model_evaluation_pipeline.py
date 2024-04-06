from farm_copilot.config.configuration import ConfigurationManager
from farm_copilot.components.model_evaluation import Evaluation
from farm_copilot.utils import logger


STAGE_NAME = "Model Evaluation"

class ModelEvaluationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.get_trained_model()
        evaluation.evaluation()
        evaluation.log_into_mlflow()



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
            