import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
def build_xception_model(input_shape=(128, 128, 3), num_classes=3):
    # Tạo base model từ Xception với weights từ ImageNet
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'  
    )
    # Chỉ fine-tune các layer cuối
    for layer in base_model.layers[:-20]:  # Freeze các layer đầu
        layer.trainable = False

    x = base_model.output
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes,
                       activation='softmax',
                       kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)


    # Compile với learning rate thấp để fine-tuning ổn định
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=5e-5, weight_decay=1e-6),  
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model

def train_xception(X_train, y_train, X_val, y_val):
    model = build_xception_model()

    # Early stopping để dừng khi validation loss không cải thiện
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,  
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
                        batch_size=32, # Có thể thử 16
                        validation_data=(X_val, y_val), 
                        callbacks=callbacks, 
                        verbose=1)
    return model, history