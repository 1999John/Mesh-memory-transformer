import tensorflow as tf
from tensorflow.python.keras.layers import Layer,LayerNormalization,Dropout,Embedding
from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.FeedForwardNetwork import point_wise_feed_forward_network
from transformer.cross_attention import CrossAttention
from transformer.utils import positional_encoding
import numpy as np

class DecoderLayer(Layer):
    def __init__(self, d_model,num_heads,num_layers,dff,rate=0.1):
        super(DecoderLayer, self).__init__()

        self.num_layers = num_layers

        self.masked_self_attention = MultiHeadAttention(num_heads,d_model)
        self.cross_attentions = [CrossAttention(d_model,num_heads,"cross_attention_{}".format(_),rate) for _ in range(num_layers)]
        self.alphas = [tf.keras.layers.Dense(d_model,activation=tf.nn.sigmoid,name="alphas_{}".format(_)) for _ in range(num_layers)]

        self.ffn = point_wise_feed_forward_network(d_model=d_model,dff=dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)


    def call(self,Y,enc_outputs,traning,
             look_ahead_mask,padding_mask):
        # enc_output (batchsize,input_seq_len,d_model)
        Q,_ = self.masked_self_attention(Y,Y,Y,look_ahead_mask)
        Q = self.dropout1(Q,traning)
        Q = self.layernorm1(Y+Q)

        M_mesh = 0

        for i in range(self.num_layers):
            ca_out = self.cross_attentions[i](Y,enc_outputs[i],padding_mask,traning)
            y_c = tf.concat([Y,ca_out],axis=-1)
            alpha_out = self.alphas[i](y_c)
            M_mesh+=alpha_out*ca_out
        M_mesh /= np.sqrt(self.num_layers)
        M_mesh = self.layernorm2(M_mesh+Y)

        ff = self.ffn(M_mesh)
        ff = self.layernorm3(M_mesh+ff)

        return ff

class Decoder(Layer):
    def __init__(self,num_layers,d_model,num_heads,dff,target_vocab_size,maximum_position_encoding,rate=0.1):
        super(Decoder,self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size,d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,d_model)

        self.dec_layers = [DecoderLayer(d_model,num_heads,num_layers,dff,rate)
                           for _ in range(num_layers)]

        self.dropout = Dropout(rate)

    def call(self,Y,enc_outputs,training,look_ahead_mask,padding_mask):
        seq_len = tf.shape(Y)[1]

        Y = self.embedding(Y)
        Y *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        Y += self.pos_encoding[:,:seq_len,:]

        Y = self.dropout(Y,training)
        for i in range(self.num_layers):
            Y = self.dec_layers[i](Y,enc_outputs,training,look_ahead_mask,padding_mask)

        return Y