import os
from tensorflow import keras
from src.farm_copilot.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        self.config = config


    def get_base_model(self):
        self.model = keras.applications.Xception(
            input_shape=tuple(self.config.params_image_size),
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes: int, freeze_all: bool, freeze_till: int, learning_rate: float, offset: int):
        if freeze_all:
            model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        
        scaled_layer = keras.layers.Rescaling(scale=1./127.5, offset=offset)(model.input)
        avg_pool = keras.layers.GlobalAveragePooling2D()(scaled_layer)
        dropout = keras.layers.Dropout(0.2)(avg_pool)
        output = keras.layers.Dense(units=classes, activation='softmax')(dropout)
        
        full_model = keras.Model(inputs=model.input,outputs= output)
        
        full_model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=len(os.listdir(Path(self.config.local_data_file))),
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate, 
            offset=self.config.params_offset
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)


    @staticmethod
    def save_model(path: Path, model: keras.Model):
        keras.saving.save_model(model=model, filepath=path)
        

