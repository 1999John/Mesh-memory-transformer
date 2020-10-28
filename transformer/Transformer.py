import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow.python.keras.models import Model
import numpy as np
from transformer.decoder import Decoder
from transformer.encoder import Encoder


class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, num_memory, dff, target_vocab_size, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, num_memory, dff, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = Dense(target_vocab_size,activation='softmax')

    def call(self, inp, tar, training, look_ahead_mask, dec_padding_mask):
        # (batch_size,num_layers,input_seq_len,d_model)
        enc_outputs = self.encoder(inp, training)
        # print("===================gap======================")
        dec_output = self.decoder(tar, enc_outputs, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output

# if __name__=='__main__':
#     def Set_GPU_Memory_Growth():
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             try:
#                     # 设置 GPU 显存占用为按需分配
#                 for gpu in gpus:
#                     tf.config.experimental.set_memory_growth(gpu, True)
#                 logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#                 print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#             except RuntimeError as e:
#                     # 异常处理
#                 print(e)
#         else:
#             print('No GPU')
#
#
#     # 放在建立模型实例之前
#     Set_GPU_Memory_Growth()
#     sample_transformer = Transformer(
#         num_layers=2,d_model=512,num_heads=8,num_memory=40,
#         dff=512,target_vocab_size=8000,pe_target=6000
#     )
#
#     temp_input = tf.random.uniform((1,2,512))
#     temp_target = tf.random.uniform((1,2))
#
#     fn_out = sample_transformer(temp_input,temp_target,training=False,
#                                 look_ahead_mask=None,
#                                 dec_padding_mask=None)
#
#     print(fn_out.shape)
#     print(sample_transformer.summary())
