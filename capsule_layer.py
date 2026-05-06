import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class CapsuleLayer(layers.Layer):

    def __init__(self, num_capsules, dim_capsule=16, routings=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.num_capsules * self.dim_capsule),
            initializer="glorot_uniform",
            trainable=True,
            name="capsule_kernel"
        )

    def call(self, inputs):
        u_hat = tf.matmul(inputs, self.W)

        u_hat = tf.reshape(
            u_hat,
            (-1, self.num_capsules, self.dim_capsule)
        )

        def squash(vectors):
            norm = tf.reduce_sum(tf.square(vectors), axis=-1, keepdims=True)
            scale = norm / (1 + norm) / tf.sqrt(norm + 1e-7)
            return scale * vectors

        v = squash(u_hat)

        return tf.norm(v, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsule": self.dim_capsule,
            "routings": self.routings
        })
        return config
