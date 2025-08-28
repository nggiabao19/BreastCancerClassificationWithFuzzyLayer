import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import clone_model
from .fuzzy_pooling import FuzzyPoolingLayer

def build_resnet50v2_fuzzyPooling_model(input_shape=(128, 128, 3), num_classes=3):
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Clone model và thay thế MaxPooling2D bằng FuzzyPoolingLayer
    def convert_layer(layer):
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            return FuzzyPoolingLayer(pool_size=layer.pool_size, strides=layer.strides, padding=layer.padding)
        return layer

    new_model = clone_model(base_model, clone_function=convert_layer)

    # Chuyển trọng số (copy weights)
    for old_layer, new_layer in zip(base_model.layers, new_model.layers):
        if isinstance(old_layer, tf.keras.layers.MaxPooling2D):
            continue  # Không chuyển trọng số cho lớp đã thay thế
        new_layer.set_weights(old_layer.get_weights())

    # Thêm các lớp classification
    x = new_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = layers.Dropout(0.3) (x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    final_model = models.Model(inputs=new_model.input, outputs=output)

    # Đóng băng các lớp base
    for layer in new_model.layers:
        layer.trainable = True

    # Biên dịch
    final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return final_model

def train_resnet50v2_fuzzyFC(X_train, y_train, X_val, y_val):
    model = build_resnet50v2_fuzzyPooling_model()

    # Early stopping để dừng khi validation loss không cải thiện
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  
        restore_best_weights=True,
        verbose=1
    )
    # Giảm learning rate khi validation loss không cải thiện
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Giảm LR xuống một nửa
        patience=8,   # Đợi 8 epochs không cải thiện
        min_lr=1e-7,
        verbose=1
    )
    callbacks = [early_stopping, reduce_lr]

    history = model.fit(X_train, y_train, 
                        epochs=200,
                        batch_size=16, 
                        validation_data=(X_val, y_val), 
                        callbacks=callbacks, 
                        verbose=1)
    return model, history