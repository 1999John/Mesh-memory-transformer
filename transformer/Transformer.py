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

        self.encoder = Encoder(num_layers, d_model, num_heads, num_memory,dff, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = Dense(target_vocab_size)

    def call(self, inp, tar, training, look_ahead_mask, dec_padding_mask):
        # (batch_size,num_layers,input_seq_len,d_model)
        enc_outputs = self.encoder(inp, training)
        # print("===================gap======================")
        dec_outputs = self.decoder(tar, enc_outputs, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_outputs)

        return final_output