from PIL import Image
import os
from src.farm_copilot.utils import logger
from src.farm_copilot.entity.config_entity import PrepareDataConfig


class PrepareData:
    def __init__(self, config: PrepareDataConfig) -> None:
        self.config = config    
    

    def resize_image(input_image_path, output_image_path, size=(224,224)):
        original_image = Image.open(input_image_path)
        resize_image = original_image.resize(size)
        resize_image.save(output_image_path)


    def is_valid_jpeg(image_path):
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except (IOError, SyntaxError):
            return False


    def clean(self):      
        valid_ext = ['.JPG', '.PNG', '.GIF', '.BMP', '.jpg', '.png', '.gif', '.bmp']


        for (root, dir, files) in os.walk(self.config.training_data):
            for file in files:
                _, ext = os.path.splitext(p=os.path.join(root, file))
                if ext not in valid_ext:
                    os.remove(os.path.join(root, file))
                    logger.info(f"{file} with extension {ext} removed. Reason: Invalid extension")
                    continue
                
                image = os.path.join(root,file)
                if not self.is_valid_jpeg(image_path=image):
                    os.remove(image)
                    logger.info(f"{image} removed. Reason: Corrupted")
                    continue
                
                try:
                    self.resize_image(input_image_path=image, output_image_path=image)
                except (IOError, SyntaxError):
                    os.remove(image)
                    logger.info(f"{image} removed. Reason: Failed to resize")