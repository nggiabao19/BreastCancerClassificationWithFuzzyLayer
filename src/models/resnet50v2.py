import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception, ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_resnet50v2_model(input_shape=(128, 128, 3), num_classes=3):
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )

    # Fine-tune: freeze nhiều hơn cho dataset nhỏ
    for layer in base_model.layers[:-10]:  # Chỉ mở 2 tầng cuối thay vì 5
        layer.trainable = False

    # Xây dựng phần classifier đơn giản hơn
    x = base_model.output
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  # Tăng dropout từ 0.3 lên 0.5
    x = layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)  # Giảm từ 32 xuống 16 neurons
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)  # Thêm một dropout nữa

    outputs = layers.Dense(num_classes,
                       activation='softmax',
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)  # Giảm regularization

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # Compile với learning rate thấp hơn
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Giảm từ 1e-4 xuống 5e-5
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_resnet50v2(X_train, y_train, X_val, y_val):
    model = build_resnet50v2_model()

    # Early stopping để dừng khi validation loss không cải thiện
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,  
        restore_best_weights=True,
        verbose=1
    )
    # Giảm learning rate khi validation loss không cải thiện
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Giảm LR xuống một nửa
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