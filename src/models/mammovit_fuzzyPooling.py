import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import clone_model
from .fuzzy_pooling import FuzzyPoolingLayer

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

def build_mammovit_fuzzyPooling_model(
    input_shape=(128, 128, 3),
    num_classes=3,
    embed_dim=256,
    num_heads=4,
    num_vit_layers=1,
    dropout_rate=0.3
):
    # 1) Tải ResNet50 gốc
    base = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 2) Hàm chuyển layer: MaxPooling2D --> FuzzyPoolingLayer
    def convert(layer):
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            return FuzzyPoolingLayer(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                name=layer.name.replace('pool', 'fuzzy_pool')
            )
        return layer

    # 3) Clone model với convert_function
    fuzzy_backbone = clone_model(
        base,
        clone_function=convert
    )

    # 4) Copy weights cho mọi layer (trừ pooling đã thay)
    for old_layer, new_layer in zip(base.layers, fuzzy_backbone.layers):
        if isinstance(old_layer, tf.keras.layers.MaxPooling2D):
            continue
        new_layer.set_weights(old_layer.get_weights())

    # 5) Tùy chỉnh trainable nếu cần
    for layer in fuzzy_backbone.layers:
        layer.trainable = True

    # 6) Gắn phần ViT lên output của backbone
    x = fuzzy_backbone.output                        # (None, H, W, C)
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    num_patches = (h * w)

    # 6.1) Patch embedding
    x = layers.Reshape((num_patches, c), name='reshape_patches')(x)
    x = layers.Dense(
        units=embed_dim,
        kernel_regularizer=regularizers.l2(2e-3),
        name='patch_proj'
    )(x)

    # 6.2) Thêm token và positional encoding
    x = ClassTokenLayer(embed_dim=embed_dim)(x)
    x = PositionalEncoding(
        num_patches=num_patches,
        embed_dim=embed_dim
    )(x)

    # 6.3) Transformer blocks (có thể fuzzy-attention bên trong)
    ff_dim = embed_dim * 4
    for i in range(num_vit_layers):
        x = LightMammoViTLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f'vit_block_{i}'
        )(x)

    # 6.4) Lấy [CLS] token để phân lớp
    x = layers.LayerNormalization(epsilon=1e-6, name='pre_cls_ln')(x)
    cls_token = layers.Lambda(lambda t: t[:, 0, :], name='extract_cls')(x)

    # 7) Fuzzy fully‑connected head
    # Thêm layer trung gian với regularization mạnh hơn
    x = layers.Dense(128, activation='gelu', 
        kernel_regularizer=regularizers.l2(0.001)  
    )(cls_token)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.8)(x) 

    outputs = layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    # 8) Đóng gói và compile
    model = models.Model(
        inputs=fuzzy_backbone.input,
        outputs=outputs,
        name='MammoViT_Fuzzy_ResNet50'
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

def train_mammovit_fuzzyPooling(X_train, y_train, X_val, y_val):
    model = build_mammovit_fuzzyPooling_model()
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