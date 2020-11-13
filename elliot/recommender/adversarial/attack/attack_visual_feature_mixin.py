import tensorflow as tf


class AttackVisualFeature:

    def set_delta(self, delta_init=0):
        """
        Set delta variables useful to store delta perturbations,
        :param delta_init: 0: zero-like initialization, 1 uniform random noise initialization
        :return:
        """
        if delta_init:
            self.delta = tf.random.uniform(shape=self.F.shape, minval=-0.05, maxval=0.05,
                                              dtype=tf.dtypes.float32, seed=0)
        else:
            self.delta = tf.Variable(tf.zeros(shape=self.F.shape), dtype=tf.dtypes.float32,
                                        trainable=False)
