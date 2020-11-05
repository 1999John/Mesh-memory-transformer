import tensorflow as tf
from transformer.utils import create_padding_mask, create_look_ahead_mask
from utils.config import Config
from utils.losses import *
from transformer.Transformer import Transformer
from utils.optimizer import *
from utils.use_cuda import Set_GPU_Memory_Growth

Set_GPU_Memory_Growth()


def create_masks(inp, tar):
    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp, flag=True)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask, dec_padding_mask


# lr = CustomSchedule(1e-6)
optimizer = tf.keras.optimizers.Adam(learning_rate=2048,
                                     beta_1=0.9,beta_2=0.98)

# def __init__(self,num_layers,d_model,num_heads,num_memory,dff,target_vocab_size,pe_target,rate=0.1):
transformer = Transformer(Config.num_layers, Config.d_model, Config.num_heads, Config.num_memory, Config.ffn,
                          Config.target_vocab_size,
                          pe_target=Config.target_vocab_size,
                          rate=Config.dropout_rate)
# transformer.load_weights("1028/new_0")

def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    # print("combined_mask.shape:{},dec_padding_mask:{}".format(combined_mask.shape,dec_padding_mask.shape))

    with tf.GradientTape() as tape:
        # def call(self,inp,tar,training,look_ahead_mask,dec_padding_mask):
        predictions = transformer(inp, tar_inp,
                                  True,
                                  combined_mask,
                                  dec_padding_mask)
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    # print(transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)


import time
from data.dataset import create_datasets_1000, get_train

train_dataset = create_datasets_1000(Config.batch_size)


try:
    for epoch in range(Config.EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            # print(transformer.trainable_variables)
            if batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            if batch % 2000==0:
                transformer.save_weights("/home/mist/Mesh/1104/epoch_{}_batch_{}".format(epoch,batch))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))
        transformer.save_weights("/home/mist/Mesh/1104/epoch_{}".format(epoch))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

except KeyboardInterrupt as k:
    transformer.save_weights("/home/mist/Mesh_memory/1028/train_k_{}".format(epoch))
