import tensorflow as tf
import numpy as np


class LOAD_SIAMESE_NETWORK :
    def W_init(self, shape, dtype=None, name=None):
        """Initialize weights as in paper"""
        values = np.random.normal(loc=0, scale=1e-2, size=shape).astype(np.float32)
        return tf.keras.backend.variable(values, dtype=tf.float32, name=name)

    def b_init(self, shape, dtype=None, name=None):
        """Initialize bias as in paper"""
        values = np.random.normal(loc=0.5, scale=1e-2, size=shape).astype(np.float32)
        return tf.keras.backend.variable(values, dtype=tf.float32, name=name)

    def create_model_siamese(self) :
        custom_objects = {'W_init': W_init, 'b_init': b_init}
        self.model = tf.keras.models.load_model('./models_folder/model_paper_71percent_ver5_ATandT_weight_of_only_siamese_network_addEuclideanDistance.h5', custom_objects=custom_objects)
        self.model.load_weights('./models_folder/paper_71percent_ver5_ATandT_weight_of_only_siamese_network_addEuclideanDistance.h5')

        return self.model