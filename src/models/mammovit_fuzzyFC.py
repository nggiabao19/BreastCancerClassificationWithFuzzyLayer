import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models, regularizers
from .fuzzy_layer import FuzzyLayer
    
class LightMammoViTLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(LightMammoViTLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Multi-head attention with reduced key_dim
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // 2,  # Reduced key_dim to lower parameters
            dropout=dropout_rate
        )

        # Smaller feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])

        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        # Multi-head attention with residual connection
        attention_output = self.attention(inputs, inputs, training=training)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config
    
class EfficientPatchEmbedding(layers.Layer):
    """
    Efficient patch embedding layer with fewer parameters
    """
    def __init__(self, patch_size, embed_dim, **kwargs):
        super(EfficientPatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Use separable convolution to reduce parameters
        self.projection = tf.keras.Sequential([
            layers.SeparableConv2D(
                filters=embed_dim // 2,
                kernel_size=patch_size,
                strides=patch_size,
                padding="valid",
                activation='relu'
            ),
            layers.Conv2D(
                filters=embed_dim,
                kernel_size=1,
                activation=None
            )
        ])

    def call(self, patch):
        # Apply projection
        patches = self.projection(patch)
        # Reshape to sequence format
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(patches, [batch_size, -1, self.embed_dim])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
        })
        return config
    
class PositionalEncoding(layers.Layer):
    """
    Positional encoding layer for Vision Transformer
    """
    def __init__(self, num_patches, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Positional embedding layer
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1,  # +1 for CLS token
            output_dim=embed_dim
        )

    def call(self, patch_embeddings):
        batch_size = tf.shape(patch_embeddings)[0]
        seq_length = tf.shape(patch_embeddings)[1]

        # Create position indices
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = self.position_embedding(positions)

        return patch_embeddings + position_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim,
        })
        return config

class ClassTokenLayer(layers.Layer):
    """
    Class token layer that adds a learnable CLS token to the sequence
    """
    def __init__(self, embed_dim, **kwargs):
        super(ClassTokenLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # Create learnable CLS token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="random_normal",
            trainable=True
        )
        super().build(input_shape)

    def call(self, patch_embeddings):
        batch_size = tf.shape(patch_embeddings)[0]

        # Tile CLS token for batch
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])

        # Concatenate CLS token with patch embeddings
        return tf.concat([cls_tokens, patch_embeddings], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
        })
        return config

def build_mammovit_fuzzyFC_model():
    inputs = layers.Input(shape=(128, 128, 3))
    embed_dim = 192  
    num_heads = 4    
    ff_dim = 768    
    num_layers = 2   
    dropout_rate = 0.1 

    # Backbone
    base = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    base.trainable = False
    # Đóng băng nhiều layer hơn để giảm overfitting
    for layer in base.layers[:-20]:  # Chỉ train 15 layer cuối thay vì 20
        layer.trainable = True

    x = base.output  # (None, 4, 4, 2048)
    num_patches = 16  # 4 * 4
    x = layers.Reshape((num_patches, 2048))(x)  # (None, 16, 2048)

    # Project patch embeddings to embed_dim
    x = layers.Dense(
        embed_dim,
        kernel_regularizer=regularizers.l2(1e-3)  
    )(x)

    # Thêm Class Token và Positional Encoding
    x = ClassTokenLayer(embed_dim)(x)
    x = PositionalEncoding(num_patches=num_patches, embed_dim=embed_dim)(x)

    # Các block Transformer
    ff_dim = embed_dim * 4 
    for _ in range(num_layers):
        x = LightMammoViTLayer(
            embed_dim,
            num_heads,
            ff_dim,
            dropout_rate
        )(x)

    # Lấy ra token đầu
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls = layers.Lambda(lambda y: y[:, 0, :])(x)

    x = layers.Dense(32, activation='gelu',  
        kernel_regularizer=regularizers.l2(0.0001)  
    )(cls)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  

    # outputs = layers.Dense(3, activation='softmax')(x)
    num_rules = 8  
    outputs = FuzzyLayer(num_rules=num_rules,
                                     num_classes=3,
                                     name='fuzzy_layer')(x)

    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=5e-5,  
        weight_decay=0.05,     
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

def train_mammovitfuzzyFC(X_train, y_train, X_val, y_val):
    model = build_mammovit_fuzzyFC_model()
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
        min_lr=1e-6,
        verbose=1
    )
    callbacks = [early_stopping, reduce_lr]

    history = model.fit(X_train, y_train, 
                        epochs=300,
                        batch_size=16, 
                        validation_data=(X_val, y_val), 
                        callbacks=callbacks, 
                        verbose=1)
    return model, history