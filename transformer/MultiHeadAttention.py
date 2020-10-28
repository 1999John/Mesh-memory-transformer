import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.python.keras.layers import Layer
from transformer.utils import *


class MeshMemoryMultiHeadAttention(Layer):
    def __init__(self,num_heads,d_model,num_memory):
        """

        :param d_model: 输出维度
        :param num_heads: 头的个数
        :param d_q: query 的维数
        :param d_k: key的维数
        :param d_v: v的维数
        :param N_memory: 记忆槽的个数
        """
        super(MeshMemoryMultiHeadAttention,self).__init__()

        self.num_heads = num_heads
        self.num_memory = num_memory
        self.d_model = d_model

        self.depth=self.d_model//self.num_heads

        self.wq = tf.keras.layers.Dense(d_model,name="memoryquery")
        self.wk = tf.keras.layers.Dense(d_model,name="memorykey")
        self.wv = tf.keras.layers.Dense(d_model,name="memoryvalue")


        self.k_memory = tf.random.normal((1,self.num_memory,d_model))
        self.v_memory = tf.random.normal((1,self.num_memory,d_model))

        self.dense = tf.keras.layers.Dense(d_model,name='memorydense')

    def split_heads(self,x,batch_size):
        """
        分拆最后一个维度到(num_heads,depth)
        转置结果使得形状为(batch_size,num_heads,seq_len,depth)
        :param x:
        :param batch_size:
        :return:
        """
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))

        return tf.transpose(x,perm=[0,2,1,3])

    def call(self,q,k,v,mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q) # (batch_size,seq_len_q,d_model)
        k = self.wk(k) # (batch_size,seq_len_k,d_model)
        v = self.wv(v) # (batch_size,seq_len_v,d_model)


        k = tf.concat([k,tf.tile(self.k_memory,[k.shape[0],1,1])],axis=1) # (batch_size,seq_len_k+num_memory,d_model)

        v = tf.concat([v,tf.tile(self.v_memory,[v.shape[0],1,1])],axis=1) # (batch_size,seq_len_q+num_memory,d_model)

        q = self.split_heads(q,batch_size) # (bsz,num_heads,seq_len_q,d_model')
        k = self.split_heads(k,batch_size) # (bsz,num_heads,seq_len_k+num_memory,d_model')
        v = self.split_heads(v,batch_size) # (bsz,num_heads,seq_len_v+num_memory,d_model')


        # scaled_attention:(batchsz,num_heads,seq_len_q,d_model')
        # attention_weights: (batchsz,num_heads,seq_len_q,seq_len_k+num_memory)
        scaled_attention,attention_weights = scaled_dot_production_attention(q,k,v,mask)

        # (batchsz,seq_len_q,num_heads,d_model')
        scaled_attention = tf.transpose(scaled_attention,perm=[0,2,1,3])

        # (batchsz,seq_len_q,d_model)
        mesh_attention = tf.reshape(scaled_attention,(batch_size,-1,self.d_model))

        mesh_attention = self.dense(mesh_attention)

        return mesh_attention,attention_weights

class MultiHeadAttention(Layer):
    def __init__(self,num_heads,d_model):
        """

        :param d_model: 输出维度
        :param num_heads: 头的个数
        :param d_q: query 的维数
        :param d_k: key的维数
        :param d_v: v的维数
        :param N_memory: 记忆槽的个数
        """
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth=self.d_model//self.num_heads

        self.wq = tf.keras.layers.Dense(d_model,name=self.name+' MUTIquery')
        self.wk = tf.keras.layers.Dense(d_model,name=self.name+' MUTIkey')
        self.wv = tf.keras.layers.Dense(d_model,name=self.name+' MUTIvalue')

        self.dense = tf.keras.layers.Dense(d_model,name=self.name+'MUTIdense')

    def split_heads(self,x,batch_size):
        """
        分拆最后一个维度到(num_heads,depth)
        转置结果使得形状为(batch_size,num_heads,seq_len,depth)
        :param x:
        :param batch_size:
        :return:
        """
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))

        return tf.transpose(x,perm=[0,2,1,3])

    def call(self,q,k,v,mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q) # (batch_size,seq_len_q,d_model)
        k = self.wk(k) # (batch_size,seq_len_k,d_model)
        v = self.wv(v) # (batch_size,seq_len_v,d_model)

        q = self.split_heads(q,batch_size) # (bsz,num_heads,seq_len_q,d_model')
        k = self.split_heads(k,batch_size) # (bsz,num_heads,seq_len_k+num_memory,d_model')
        v = self.split_heads(v,batch_size) # (bsz,num_heads,seq_len_v+num_memory,d_model')


        # scaled_attention:(batchsz,num_heads,seq_len_q,d_model')
        # attention_weights: (batchsz,num_heads,seq_len_q,seq_len_k+num_memory)
        scaled_attention,attention_weights = scaled_dot_production_attention(q,k,v,mask)

        # (batchsz,seq_len_q,num_heads,d_model')
        scaled_attention = tf.transpose(scaled_attention,perm=[0,2,1,3])

        # (batchsz,seq_len_q,d_model)
        mesh_attention = tf.reshape(scaled_attention,(batch_size,-1,self.d_model))

        mesh_attention = self.dense(mesh_attention)

        return mesh_attention,attention_weights



