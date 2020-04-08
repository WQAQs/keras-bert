import os
import numpy as np
import tensorflow as tf
from keras_bert.backend import keras
from keras_bert.backend import backend as K
from keras_bert.layers import get_inputs, get_embedding, TokenEmbedding, EmbeddingSimilarity, Masked, Extract, TaskEmbedding
from keras_bert import (get_model, compile_model, get_base_dict, gen_batch_inputs, get_token_embedding,
                        get_custom_objects, set_custom_objects, loader)
from indoor_location.utils import (get_sentence_pairs, load_dataset, gen_bert_data)
from keras_bert.layers import get_inputs
from keras_bert import load_trained_model_from_checkpoint,build_model_from_config

seqence_len = 26  #有效的ap数量
pretrain_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
train_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
test_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
flag_retrain = True
MODEL_DIR = "..\\model\\"
pretrained_model_path = MODEL_DIR + "pretrained_bert1.h5"
trained_model_path = MODEL_DIR + "trained_bert1.h5"

pretrained_path = '.\\logs'
config_path = os.path.join(pretrained_path, 'mybert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'mybert_model.ckpt')
# config_path = 'mybert_config.json'
# checkpoint_path = 'mybert_model.ckpt'
flag_retrain = True
LR = 0.001
EPOCHS = 10
BATCH_SIZE = 128

def bert_indoorlocation_train_with_label():
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True

    # 准备训练集数据和验证集数据
    sentence_pairs = get_sentence_pairs(pretrain_datafile_name)
    token_dict = get_base_dict()
    for pairs in sentence_pairs:
        for token in pairs[0] + pairs[1]:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    token_list = list(token_dict.keys())

    # x_train, y_train = gen_batch_inputs(
    #     sentence_pairs,
    #     token_dict,
    #     token_list,
    #     seq_len=seqence_len,
    #     mask_rate=0.3,
    #     swap_sentence_rate=1.0,
    # )

    x_train, y_train = gen_bert_data(train_datafile_name, seqence_len)
    x_test, y_test = gen_bert_data(test_datafile_name, seqence_len)

    # x_train, y_train = np.array(x_train), np.array(y_train)
    # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    # y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    # with tf.Session(config=config) as sess:

    # model = load_trained_model_from_checkpoint(
    #     config_path,
    #     checkpoint_path,
    #     training=False,  ###！！！十分重要！！！！
    #     trainable=True,
    #     seq_len=seqence_len,
    # )
    # 初始化模型和参数
    # mymodel, myconfig = build_model_from_config(
    #     config_path,
    #     training=False,
    #     trainable=True)
    input_layer, transformed = get_model(
        token_num=len(token_dict),
        head_num=2,
        transformer_num=2,
        embed_dim=12,
        feed_forward_dim=100,
        seq_len=seqence_len,
        pos_num=seqence_len,
        dropout_rate=0.05,
        attention_activation='gelu',
        training=False,
        trainable=True
    )
    # output_layer = model.inputs[:2]
    # dense = model.get_layer('Encoder-2-FeedForward-Norm').output
    # output_layer = keras.layers.Dense(units=2, activation='relu')(dense)
    extract_layer = Extract(index=0, name='Extract')(transformed)
    # coor_dense = keras.layers.Dense(units=embed_dim, activation="relu", name="coor_dense")(transformed)
    output_layer = keras.layers.Dense(units=2, activation="relu", name="coor_output")(extract_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.load_weights(pretrained_model_path, by_name=True)
    optimizer = keras.optimizers.RMSprop(LR)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse'],
    )
    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    model.save(trained_model_path)
    #
    # predicts = model.predict(x_train)
    # train_y = list(map(lambda x: np.squeeze(x, axis=-1), y_train))
    # predicts = list(map(lambda x: np.argmax(x, axis=-1), predicts))
    # predicts

bert_indoorlocation_train_with_label()

