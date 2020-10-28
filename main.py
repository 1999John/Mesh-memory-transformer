"""
author: 徐志
time:2020/10/25
QQ:1808212297
"""

from transformer.Transformer import Transformer
import tensorflow as tf
import tensorflow.python.keras as keras


def Set_GPU_Memory_Growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置 GPU 显存占用为按需分配
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # 异常处理
            print(e)
    else:
        print('No GPU')


if __name__ == '__main__':
    Set_GPU_Memory_Growth()
    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, num_memory=40,
        dff=512, target_vocab_size=8000, pe_target=6000
    )

    temp_input = tf.random.uniform((1, 2, 512))
    temp_target = tf.random.uniform((1, 2))

    fn_out = sample_transformer(temp_input, temp_target, training=False,
                                look_ahead_mask=None,
                                dec_padding_mask=None)

    print(fn_out.shape)
    print(sample_transformer.summary())