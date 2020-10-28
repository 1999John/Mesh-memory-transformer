import tensorflow as tf
import numpy as np

def get_angles(pos,i,d_model):
    '''
    :param pos:np.array,shape:[possition,1]
    :param i:np.array,shape:[1,d_model]
    :param d_model:np.uint
    :return: np.array
    '''
    angle_rate = 1/np.power(10000,(2*(i//2))/np.float32(d_model))
    return pos*angle_rate
def positional_encoding(position,d_model):
    angle_rads = get_angles(np.arange(position)[:,np.newaxis],
                            np.arange(d_model)[np.newaxis,:],
                            d_model)
    angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])
    angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])

    pos_encoding = angle_rads[np.newaxis,...]

    return tf.cast(pos_encoding,dtype=tf.float32)


def create_padding_mask(seq,flag=False):
    """
    flag 为true表示图片的遮挡
    遮挡一批序列中所有的填充标记，确保模型不会将填充作为输入。该mask表明填充值为0出现的位置：在这些位置输出1
    :param seq: (batch_size,seq_len,d_model)
    :return: (batch_size,1,1,seq_len)
    """
    if flag:
        seq = tf.cast(tf.math.equal(seq,tf.zeros([2048],seq.dtype))[...,0],tf.float32)
    else:
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:,tf.newaxis,tf.newaxis,:]

def create_look_ahead_mask(size):
    """
    创建上三角矩阵，意味着，预测第i个词只需要前i-1个词
    :param size:
    :return:
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size,size)),-1,0)
    return mask

def scaled_dot_production_attention(q,k,v,mask):
    """

    :param q: (batchsz,num_heads,seq_len_q,d_q)
    :param k: (bsz,num_heads,seq_len_k+num_memory,d_k)
    :param v: (bsz,num_heads,seq_len_v+num_memory,d_v)
    :param mask: [batchsz,seq_len_q,seq_len_k) default:None
    :return:
    """
    matmul_qk = tf.matmul(q,k,transpose_b=True) # [batchsz,num_heads,seq_len_q,seq_len_k+num_memory]

    # 缩放
    dk = tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)

    # 将非None的mask加入到微缩的张量上
    if mask is not None:
        scaled_attention_logits += (mask*-1e9)

    # softmax 在最后一个轴归一化，即在seq_len_k所在轴归一化
    # (batchsz,num_heads,seq_len_q,seq_len_k+num_memory)
    attention_weights = tf.nn.softmax(scaled_attention_logits,axis=-1)

    output = tf.matmul(attention_weights,v) # (batchsz,num_heads,seq_len_q,d_v)

    return output,attention_weights