import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Layer,Dropout
from MultiHeadAttention import MultiHeadAttention

class CrossAttention(Layer):
    def __init__(self,d_model,num_heads,name,rate=0.1):
        super(CrossAttention,self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.dropout = Dropout(rate)
    def call(self,Y,X,mask,training):
        out,_ = self.mha(Y,X,X,mask)
        out = self.dropout(out,training)

        return out