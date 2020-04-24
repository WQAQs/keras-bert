import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_bert.backend import keras
from keras_bert.backend import backend as K
from keras_bert.layers import get_inputs, get_embedding, TokenEmbedding, EmbeddingSimilarity, Masked, Extract, TaskEmbedding
from keras_bert import (get_model, compile_model, get_base_dict, gen_batch_inputs, get_token_embedding,
                        get_custom_objects, set_custom_objects, loader)
# from indoor_location.utils import (get_sentence_pairs, gen_bert_data, evaluate_fine_tune_model)
from indoor_location import utils
from keras_bert.layers import get_inputs
from keras_bert import load_trained_model_from_checkpoint,build_model_from_config
import keras_bert
from indoor_location import globalConfig
from indoor_location import hyper_parameters as hp


# seqence_len = 26  #有效的ap数量
valid_ibeacon_num = 26 #有效的ap数量
# seqence_len = valid_ibeacon_num*2+3   # 因为tokens = [TOKEN_CLS] + first + [TOKEN_SEP] + second + [TOKEN_SEP]
seq_len = hp.seq_len

# pretrain_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
# train_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
# test_datafile_name = "..\\data\\sampleset_data\\sampleset_1days_21points_average_interval_500ms.csv"

pretrain_train_datafile_path = "./data/sampleset_data/new_3days1/pretrain_dataset.csv"
train_datafile_path = "./data/sampleset_data/new_3days1/valid_dataset1.csv"
valid_datafile_path = ".\\data\\sampleset_data\\7days\\valid_dataset1.csv"
test_datafile_path = "./data/sampleset_data/new_3days2/train_dataset1.csv"
# \\1days\\sampleset_1days_20points_mac_rssi_word.csv
# token_id_from_numerical_order_file_path = ".\\logs\\token_id_from_numerical_order.csv"
# token_id_from_dataset_order_file_path = ".\\logs\\token_id_from_dataset__order1.csv"
word_id_map_file_path = globalConfig.word_id_map_file_path

# MODEL_DIR = ".\\logs\\"
# experiment_index = "new1"
# trained_model_index = "5"
# pretrained_model_index = "5"
pretrained_model_path = ".\\logs\\pretrained_bert2.h5"
trained_model_path = ".\\logs\\trained1_bert.h5"
# all_mask_real_tokens_file_path = ".\\logs\\all_mask_real_tokens.csv"
# all_mask_predict_tokens_file_path = ".\\logs\\all_mask_predict_tokens.csv"
# all_match_res_file_path = ".\\logs\\all_match_res.csv"

# pretrained_path = '.\\logs'
# config_path = os.path.join(pretrained_path, 'mybert_config.json')
# checkpoint_path = os.path.join(pretrained_path, 'mybert_model.ckpt')
# config_path = 'mybert_config.json'
# checkpoint_path = 'mybert_model.ckpt'

flag_retrain = False
only_evaluate_history_model_flag = False
LR = 0.05
EPOCHS = 100
BATCH_SIZE = 128

TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking

def get_base_id2token_dict():
    return {
        0: TOKEN_PAD,
        1: TOKEN_UNK,
        2: TOKEN_CLS,
        3: TOKEN_SEP,
        4: TOKEN_MASK,
    }

def get_finetune_model():
    word2id_dict, id2word_dict = utils.get_word_id_map(word_id_map_file_path)
    input_layer, transformed = keras_bert.my_get_model(
        token_num=len(word2id_dict),
        head_num=hp.head_num,
        transformer_num=hp.transformer_num,
        embed_dim=hp.embed_dim,
        feed_forward_dim=hp.feed_forward_dim,
        dropout_rate=hp.dropout_rate,
        seq_len=hp.seq_len,
        pos_num=hp.seq_len,
        attention_activation='gelu',
        training=False,  ### !!!!!!!一定不能忘记设置为False！！！！！！！！！
        trainable=True
    )
    # output_layer = model.inputs[:2]
    # dense = model.get_layer('Encoder-2-FeedForward-Norm').output
    # output_layer = keras.layers.Dense(units=2, activation='relu')(dense)
    extract_layer = Extract(index=0, name='Extract')(transformed)
    # coor_dense = keras.layers.Dense(units=embed_dim, activation="relu", name="coor_dense")(transformed)
    output_layer = keras.layers.Dense(units=2, activation="relu", name="coor_output")(extract_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model

def run_experiment_different_label_rate():
    label_rate = [0.005, 0.01, 0.1, 0.5, 0.9, 1.0]
    dist_dir = '/'.join(pretrain_train_datafile_path.split('/')[:-1])  #得到上一级目录
    if not os.path.exists(dist_dir+'/label_rate'):
        os.makedirs(dist_dir+'/label_rate')
    model = get_finetune_model()
    if flag_retrain or only_evaluate_history_model_flag:
        model.load_weights(trained_model_path)
    else:
        model.load_weights(pretrained_model_path, by_name=True)
    model.summary()
    optimizer = keras.optimizers.RMSprop(LR)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse'],
    )
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=5)
    for x in label_rate:
        str1 = '_'.join(str(x).split('.'))
        target_label_file = '/'.join([dist_dir, str1+'.csv'])
        utils.gen_label_rate_dataset_file(pretrain_train_datafile_path, x, target_label_file)
        x_train, y_train, reference_tags_train = utils.gen_fine_tune_bert_data(target_label_file, seq_len)
        results_dir = '/'.join(['./logs/record_results', str1])
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if only_evaluate_history_model_flag:
            utils.evaluate_fine_tune_model(model, test_datafile_path,results_dir)
        else:
            model.fit(
                x_train,
                y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stopping]
            )
            model.save(trained_model_path)
            utils.evaluate_fine_tune_model(model, test_datafile_path,results_dir)

def bert_indoorlocation_train_with_label():
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True

    # 准备训练集数据和验证集数据
    word2id_dict, id2word_dict = utils.get_word_id_map(word_id_map_file_path)
    x_train, y_train, reference_tags_train = utils.gen_fine_tune_bert_data(train_datafile_path, seq_len)
    x_valid, y_valid, reference_tags_valid = utils.gen_fine_tune_bert_data(valid_datafile_path, seq_len)
    # x_test, y_test, reference_tags_test = utils.gen_fine_tune_bert_data(test_datafile_name, seq_len)

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
    input_layer, transformed = keras_bert.my_get_model(
        token_num=len(word2id_dict),
        head_num=hp.head_num,
        transformer_num=hp.transformer_num,
        embed_dim=hp.embed_dim,
        feed_forward_dim=hp.feed_forward_dim,
        dropout_rate=hp.dropout_rate,
        seq_len=hp.seq_len,
        pos_num=hp.seq_len,
        attention_activation='gelu',
        training=False,   ### !!!!!!!一定不能忘记设置为False！！！！！！！！！
        trainable=True
    )
    # output_layer = model.inputs[:2]
    # dense = model.get_layer('Encoder-2-FeedForward-Norm').output
    # output_layer = keras.layers.Dense(units=2, activation='relu')(dense)
    extract_layer = Extract(index=0, name='Extract')(transformed)
    # coor_dense = keras.layers.Dense(units=embed_dim, activation="relu", name="coor_dense")(transformed)
    output_layer = keras.layers.Dense(units=2, activation="relu", name="coor_output")(extract_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    if flag_retrain or only_evaluate_history_model_flag:
        model.load_weights(trained_model_path)
    else:
        model.load_weights(pretrained_model_path, by_name=True)
    model.summary()

    if not only_evaluate_history_model_flag:
        optimizer = keras.optimizers.RMSprop(LR)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse'],
        )
        early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=5)
        # model.fit(
        #     x_train,
        #     y_train,
        #     validation_data=(x_valid, y_valid),
        #     epochs=EPOCHS,
        #     batch_size=BATCH_SIZE,
        #     callbacks=[early_stopping]
        # )
        model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping]
        )
        model.save(trained_model_path)

        # predicts = model.predict(x_train)
        # labels = y_train
        # reference_tags = reference_tags_train
        # evaluate_fine_tune_model(predicts, labels, reference_tags)

        # predicts = model.predict(x_test)
        # labels = y_test
        # reference_tags = reference_tags_test
        # utils.evaluate_fine_tune_model(predicts, labels, reference_tags)
        utils.evaluate_fine_tune_model(model, test_datafile_path)
    else:
        # predicts = model.predict(x_test)
        # labels = y_test
        # reference_tags = reference_tags_test
        # utils.evaluate_fine_tune_model(predicts, labels, reference_tags)
        utils.evaluate_fine_tune_model(model, test_datafile_path)

# bert_indoorlocation_train_with_label()

run_experiment_different_label_rate()
