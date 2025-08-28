import tensorflow as tf


class FuzzyPoolingLayer(tf.keras.layers.Layer):
    """
    Fuzzy Pooling Layer thay thế cho MaxPooling hoặc AveragePooling
    Sử dụng fuzzy logic để quyết định tầm quan trọng của mỗi pixel
    """
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super(FuzzyPoolingLayer, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding.upper()
        self.epsilon = 1e-6
        
    def build(self, input_shape):
        # Số channels của input
        self.channels = input_shape[-1]
        
        # Tham số cho fuzzy membership functions
        # Mỗi channel có các tham số riêng cho fuzzy logic
        self.alpha = self.add_weight(
            name='alpha',
            shape=(self.channels,),
            initializer=tf.keras.initializers.Constant(1.0),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True
        )
        
        self.beta = self.add_weight(
            name='beta', 
            shape=(self.channels,),
            initializer=tf.keras.initializers.Constant(0.5),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=2.0),
            trainable=True
        )
        
        # Weights để kết hợp fuzzy max và fuzzy average
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.channels,),
            initializer=tf.keras.initializers.Constant(0.5),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0),
            trainable=True
        )
        
        super(FuzzyPoolingLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Extract patches từ input tensor
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )
        
        # Reshape patches để có shape [batch, height, width, patch_size*patch_size, channels]
        batch_size = tf.shape(inputs)[0]
        patch_height = tf.shape(patches)[1]
        patch_width = tf.shape(patches)[2]
        patch_size = self.pool_size[0] * self.pool_size[1]
        
        patches = tf.reshape(patches, [batch_size, patch_height, patch_width, patch_size, self.channels])
        
        # Fuzzy membership calculation
        # Tính membership degree dựa trên giá trị pixel
        alpha_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.alpha, 0), 0), 0)
        beta_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.beta, 0), 0), 0)
        
        # Membership function: sigmoid-based
        membership = tf.nn.sigmoid(alpha_exp * (patches - beta_exp))
        
        # Fuzzy max pooling
        fuzzy_max = tf.reduce_max(patches * membership, axis=3)
        
        # Fuzzy average pooling  
        weighted_sum = tf.reduce_sum(patches * membership, axis=3)
        membership_sum = tf.reduce_sum(membership, axis=3) + self.epsilon
        fuzzy_avg = weighted_sum / membership_sum
        
        # Combine fuzzy max and fuzzy average
        gamma_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.gamma, 0), 0), 0)
        output = gamma_exp * fuzzy_max + (1 - gamma_exp) * fuzzy_avg
        
        return output
    
    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            out_height = (input_shape[1] - self.pool_size[0]) // self.strides[0] + 1
            out_width = (input_shape[2] - self.pool_size[1]) // self.strides[1] + 1
        else:  # SAME
            out_height = input_shape[1] // self.strides[0]
            out_width = input_shape[2] // self.strides[1]
        
        return (input_shape[0], out_height, out_width, input_shape[3])


