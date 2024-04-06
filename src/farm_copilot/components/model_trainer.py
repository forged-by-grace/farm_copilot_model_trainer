from tensorflow import keras
from farm_copilot.entity.config_entity import TrainingConfig
from pathlib import Path
import tensorflow as tf


class Training:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
    
    
    def get_base_model(self):
        self.model = keras.models.load_model(filepath=self.config.updated_base_model_path)

    
    def load_datasets(self):
        self.train_ds = keras.preprocessing.image_dataset_from_directory(
                        directory=self.config.training_ds_path,
                        shuffle=True,
                        label_mode='categorical',
                        image_size = tuple(self.config.params_image_size), 
                        batch_size = self.config.params_batch_size,
                        )
        
        self.val_ds = keras.preprocessing.image_dataset_from_directory(
                        directory=self.config.validation_ds_path,
                        shuffle=True,
                        label_mode='categorical',
                        image_size = tuple(self.config.params_image_size), 
                        batch_size = self.config.params_batch_size,
                        ) 
    

    def data_augumentation(self):
        augmentation_layers = keras.Sequential([
            keras.layers.RandomFlip('horizontal_and_vertical'),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2),            
            keras.layers.RandomContrast(0.2)
        ])
        
        # Augument the training set
        self.train_ds = self.train_ds.map(lambda x, y: (augmentation_layers(x), y))


    def prefetch(self):
        # Preload images into memory using combined CPU and GPU
        self.train_ds = self.train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


    def train(self):        
        self.model.fit(
            self.train_ds,
            epochs=self.config.params_epochs,            
            validation_data=self.val_ds
        )


        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        
    @staticmethod
    def save_model(path: Path, model: keras.Model):
        keras.models.save_model(model=model, filepath=path)
