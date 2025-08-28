import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .fuzzy_layer import FuzzyLayer
    
def build_resnet50v2_fuzzyFC_model(input_shape=(128, 128, 3), num_classes=3):
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )

    x = base_model.output
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Thêm một lớp Dense để giảm chiều dữ liệu trước khi vào FuzzyLayer
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)

    num_rules = 12  
    outputs = FuzzyLayer(num_rules=num_rules,
                                     num_classes=3,
                                     name='fuzzy_layer')(x)

    # Tạo mô hình
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # Chú ý: Sử dụng from_logits=True do FuzzyLayer không có activation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    return model

def train_resnet50v2_fuzzyFC(X_train, y_train, X_val, y_val):
    model = build_resnet50v2_fuzzyFC_model()

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
                        batch_size=32, 
                        validation_data=(X_val, y_val), 
                        callbacks=callbacks, 
                        verbose=1)
    return model, history