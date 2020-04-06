import os
import numpy as np
import tensorflow as tf
from keras_bert.backend import keras
from keras_bert.backend import backend as K
from keras_bert import (get_model, compile_model, get_base_dict, gen_batch_inputs, get_token_embedding,
                        get_custom_objects, set_custom_objects, mytest_get_model,loader)
from indoor_location.utils import (get_sentence_pairs,h5_model_file2json_model_file,load_dataset)
from keras_bert.layers import get_inputs
from keras_bert import load_trained_model_from_checkpoint

seqence_len = 26  #有效的ap数量
pretrain_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
train_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
flag_retrain = True
MODEL_DIR = "..\\model\\"
pretrained_model_path = MODEL_DIR + "pretrained_bert1.h5"
trained_model_path = MODEL_DIR + "trained_bert1.h5"

pretrained_path = '.\\logs'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
# config_path = 'mybert_config.json'
# checkpoint_path = 'mybert_model.ckpt'
flag_retrain = True
LR = 0.001
EPOCHS = 10
BATCH_SIZE = 128

def indoorlocation_train_bert_with_label():
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

    x_train, y_train = gen_batch_inputs(
        sentence_pairs,
        token_dict,
        token_list,
        seq_len=seqence_len,
        mask_rate=0.3,
        swap_sentence_rate=1.0,
    )
    x_test, y_test = gen_batch_inputs(
        sentence_pairs,
        token_dict,
        token_list,
        seq_len=seqence_len,
        mask_rate=0.3,
        swap_sentence_rate=1.0,
    )

    # with tf.Session(config=config) as sess:

    model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        training=False,
        trainable=True,
        seq_len=seqence_len,
    )
    # inputs = model.inputs[:2]
    # dense = model.get_layer('NSP-Dense').output
    # outputs = keras.layers.Dense(units=2, activation='softmax')(dense)
    #
    # model = keras.models.Model(inputs, outputs)
    #
    # optimizer = keras.optimizers.RMSprop(LR)
    # model.compile(
    #     optimizer=optimizer,
    #     loss='mse',
    #     metrics=['mae', 'mse'],
    # )
    model.summary()

    train_x, train_y, train_reference_tags = load_dataset(train_datafile_name)


    # model.fit(
    #     train_x,
    #     train_y,
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    # )
    # model.save(model_path)

    # predicts = model.predict(train_x)
    # train_y = list(map(lambda x: np.squeeze(x, axis=-1), train_y))
    # predicts = list(map(lambda x: np.argmax(x, axis=-1), predicts))
    # predicts

indoorlocation_train_bert_with_label()

