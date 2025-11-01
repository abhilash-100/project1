import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

class SqueezeExcitation(layers.Layer):
    def __init__(self, filters, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(self.filters // self.ratio, activation='relu')
        self.dense2 = layers.Dense(self.filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, self.filters))
        self.multiply = layers.Multiply()

    def call(self, inputs):
        se = self.global_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)
        return self.multiply([inputs, se])

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio
        })
        return config


def inception_block(x, filters_1x1, filters_3x3_reduce, filters_3x3,
                     filters_5x5_reduce, filters_5x5, filters_pool_proj):
    branch1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    branch2 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    branch2 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(branch2)

    branch3 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    branch3 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(branch3)

    branch4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch4 = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(branch4)

    output = layers.Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    return output


def build_hssan_model(input_shape=(224, 224, 150), num_classes=102):
    inputs = layers.Input(shape=input_shape)

    x = layers.Reshape((input_shape[0], input_shape[1], input_shape[2], 1))(inputs)

    x = layers.Conv3D(16, (5, 5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    shape = x.shape
    new_height = shape[1]
    new_width = shape[2]
    new_channels = shape[3] * shape[4]
    x = layers.Reshape((new_height, new_width, new_channels))(x)

    x = inception_block(x,
                        filters_1x1=64,
                        filters_3x3_reduce=64,
                        filters_3x3=128,
                        filters_5x5_reduce=32,
                        filters_5x5=64,
                        filters_pool_proj=32)

    total_filters = 64 + 128 + 64 + 32
    x = SqueezeExcitation(total_filters)(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = inception_block(x,
                        filters_1x1=128,
                        filters_3x3_reduce=96,
                        filters_3x3=192,
                        filters_5x5_reduce=48,
                        filters_5x5=96,
                        filters_pool_proj=64)

    total_filters = 128 + 192 + 96 + 64
    x = SqueezeExcitation(total_filters)(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='HSSAN')

    return model


def build_hssan_rgb_model(num_classes=102):
    inputs = layers.Input(shape=(224, 224, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = inception_block(x,
                        filters_1x1=64,
                        filters_3x3_reduce=64,
                        filters_3x3=128,
                        filters_5x5_reduce=32,
                        filters_5x5=64,
                        filters_pool_proj=32)

    total_filters = 64 + 128 + 64 + 32
    x = SqueezeExcitation(total_filters)(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = inception_block(x,
                        filters_1x1=128,
                        filters_3x3_reduce=96,
                        filters_3x3=192,
                        filters_5x5_reduce=48,
                        filters_5x5=96,
                        filters_pool_proj=64)

    total_filters = 128 + 192 + 96 + 64
    x = SqueezeExcitation(total_filters)(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='HSSAN_RGB')

    return model


if __name__ == '__main__':
    model = build_hssan_rgb_model(num_classes=102)
    model.summary()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.save('hssan_rgb_untrained.h5')
    print("Model architecture saved successfully!")
