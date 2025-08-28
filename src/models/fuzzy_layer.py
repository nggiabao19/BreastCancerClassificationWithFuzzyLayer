import tensorflow as tf


class FuzzyLayer(tf.keras.layers.Layer):
    def __init__(self, num_rules, num_classes, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.num_rules = num_rules
        self.num_classes = num_classes
        self.epsilon = 1e-6

    def build(self, input_shape):
        # Initialize centers with better scaling
        self.mu = self.add_weight(
            name='mu',
            shape=(input_shape[-1], self.num_rules),
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.1),
            trainable=True
        )

        # Initialize widths with constraint to prevent too small values
        self.sigma = self.add_weight(
            name='sigma',
            shape=(input_shape[-1], self.num_rules),
            initializer=tf.keras.initializers.Constant(1.0),
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=0.1, max_value=10.0, rate=1.0
            ),
            trainable=True
        )

        # Rule weights with regularization
        self.rule_weights = self.add_weight(
            name='rule_weights',
            shape=(self.num_rules, self.num_classes),
            initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=tf.keras.regularizers.l2(1e-3),
            trainable=True
        )
        super(FuzzyLayer, self).build(input_shape)

    def call(self, inputs):
        # Gaussian membership with numerical stability
        inputs_exp = tf.expand_dims(inputs, -1)  # [batch, features, 1]
        inputs_tiled = tf.tile(inputs_exp, [1, 1, self.num_rules])

        # Safe division and exponent
        sigma_safe = tf.nn.softplus(self.sigma) + self.epsilon
        diff = inputs_tiled - self.mu
        membership = tf.exp(-tf.square(diff) / (2 * tf.square(sigma_safe)))

        # Rule strength with log-sum-exp trick for stability
        log_membership = -tf.square(diff) / (2 * tf.square(sigma_safe))
        log_rule_strength = tf.reduce_sum(log_membership, axis=1)
        rule_strength = tf.exp(log_rule_strength -
                             tf.reduce_max(log_rule_strength, axis=-1, keepdims=True))

        # Normalized rules
        rule_sum = tf.reduce_sum(rule_strength, axis=-1, keepdims=True) + self.epsilon
        normalized_rules = rule_strength / rule_sum

        # Final output with temperature scaling
        logits = tf.matmul(normalized_rules, self.rule_weights)
        return tf.nn.softmax(logits / 0.1)  # Temperature of 0.1


