from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import tensorflow as tf

class CustomSchedule(LearningRateSchedule):
    def __init__(self,d_model,warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model,tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self,step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step*(self.warmup_steps ** -1.5)

        return tf.math.sqrt(arg1)*tf.math.minimum(arg1,arg2)

if __name__=='__main__':
    lr = CustomSchedule(1024)
    optimizer = tf.keras.optimizers.Adam(lr,beta_1=0.9,beta_2=0.98,epsilon=1e-9)

    import matplotlib.pyplot as plt

    plt.plot(lr(tf.range(40000,dtype=tf.float32)))
    plt.show()