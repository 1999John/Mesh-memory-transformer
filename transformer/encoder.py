import tensorflow as tf
from tensorflow.python.keras.layers import Layer, LayerNormalization, Dropout
from transformer.MultiHeadAttention import *
from transformer.FeedForwardNetwork import point_wise_feed_forward_network


class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, num_memory, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MeshMemoryMultiHeadAttention(d_model=d_model, num_heads=num_heads, num_memory=num_memory)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        mesh_attn_output, _ = self.mha(x, x, x,None)  # (batch_size,input_seq_len,d_model)
        mesh_attn_output = self.dropout1(mesh_attn_output, training=training)
        out1 = self.layernorm1(x + mesh_attn_output)

        ffn_output = self.ffn(out1)  # (batch_size,input_seq_len,d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class Encoder(Layer):
    def __init__(self,num_layers,d_model,num_heads,num_memory,dff,
                 rate=0.1):
        super(Encoder,self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_memory = num_memory

        self.enc_layers = [EncoderLayer(d_model,num_heads,num_memory,dff,rate) for _ in range(num_layers)]


    def call(self,x,training):
        ret = []
        for i in range(self.num_layers):
            x = self.enc_layers[i](x,training)
            ret.append(x)
        return tf.cast(ret,tf.float32)
