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

# seqence_len = 26  #有效的ap数量
valid_ibeacon_num = 26 #有效的ap数量
# seqence_len = valid_ibeacon_num*2+3   # 因为tokens = [TOKEN_CLS] + first + [TOKEN_SEP] + second + [TOKEN_SEP]
seqence_len = valid_ibeacon_num + 1  # 因为tokens = [TOKEN_CLS] + first
pretrain_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
# train_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
# test_datafile_name = "..\\data\\sampleset_data\\sampleset_1days_21points_average_interval_500ms.csv"

train_datafile_name = "..\\data\\sampleset_data\\train_dataset1.csv"
valid_datafile_name = "..\\data\\sampleset_data\\valid_dataset1.csv"
test_datafile_name = "..\\data\\sampleset_data\\valid_dataset1.csv"

token_id_from_numerical_order_file_path = ".\\logs\\token_id_from_numerical_order.csv"
token_id_from_dataset_order_file_path = ".\\logs\\token_id_from_dataset__order1.csv"

# MODEL_DIR = ".\\logs\\"
experiment_index = "6"
trained_model_index = "6"
pretrained_model_index = "4"
pretrained_model_path = ".\\logs\\pretrained_bert" + pretrained_model_index + ".h5"
trained_model_path = ".\\logs\\trained_bert" + trained_model_index + ".h5"
all_mask_real_tokens_file_path = ".\\logs\\all_mask_real_tokens" + experiment_index + ".csv"
all_mask_predict_tokens_file_path = ".\\logs\\all_mask_predict_tokens" + experiment_index + ".csv"
all_match_res_file_path = ".\\logs\\all_match_res" + experiment_index + ".csv"

# pretrained_path = '.\\logs'
# config_path = os.path.join(pretrained_path, 'mybert_config.json')
# checkpoint_path = os.path.join(pretrained_path, 'mybert_model.ckpt')
# config_path = 'mybert_config.json'
# checkpoint_path = 'mybert_model.ckpt'
flag_retrain = False
only_evaluate_history_model_flag = False
LR = 0.001
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

def bert_indoorlocation_train_with_label():
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True

    # 准备训练集数据和验证集数据
    # sentence_pairs = get_sentence_pairs(pretrain_datafile_name)
    # if not os.path.exists(token_id_from_numerical_order_file_path):
    #     dataset_token_dict = get_base_dict()  # dataset_token_dict：key为token，value为token对应的id
    #     id_dict = get_base_id2token_dict()  # id_dict：key为id，value为id对应的token
    #     df_data = [[TOKEN_PAD, 0],
    #                [TOKEN_UNK, 1],
    #                [TOKEN_CLS, 2],
    #                [TOKEN_SEP, 3],
    #                [TOKEN_MASK, 4]
    #     ]
    #     base_tokens = [TOKEN_PAD, TOKEN_UNK, TOKEN_CLS, TOKEN_SEP, TOKEN_MASK]
    #     other_tokens = []
    #
    #     for pairs in sentence_pairs:
    #         for token in pairs[0] + pairs[1]:
    #             if token not in dataset_token_dict:
    #                 n = len(dataset_token_dict)
    #                 dataset_token_dict[token] = n
    #                 other_tokens.append(token)
    #                 id_dict[n] = token
    #                 df_data.append([token, n])
    #     other_tokens = sorted(other_tokens, key=lambda x: int(x))
    #     all_tokens = base_tokens + other_tokens
    #     all_ids = [i for i in range(len(all_tokens))]
    #     pd.DataFrame(data={"token": all_tokens, "id": all_ids}).to_csv(token_id_from_numerical_order_file_path)
    #     pd.DataFrame(df_data, columns=["token", "id"]).to_csv(token_id_from_dataset_order_file_path)

    df_data = pd.read_csv(token_id_from_numerical_order_file_path)
    tokens = df_data["token"]
    ids = df_data["id"]
    token_dict = dict(zip(tokens, ids))
    id_dict = dict(zip(ids, tokens))

    token_list = list(token_dict.keys())

    # x_train, y_train = gen_batch_inputs(
    #     sentence_pairs,
    #     token_dict,
    #     token_list,
    #     seq_len=seqence_len,
    #     mask_rate=0.3,
    #     swap_sentence_rate=1.0,
    # )

    x_train, y_train, reference_tags_train = utils.gen_fine_tune_bert_data(train_datafile_name, seqence_len)
    x_valid, y_valid,reference_tags_valid = utils.gen_fine_tune_bert_data(valid_datafile_name,seqence_len)
    x_test, y_test, reference_tags_test = utils.gen_fine_tune_bert_data(test_datafile_name, seqence_len)

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
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping]
        )
        model.save(trained_model_path)

        # predicts = model.predict(x_train)
        # labels = y_train
        # reference_tags = reference_tags_train
        # evaluate_fine_tune_model(predicts, labels, reference_tags)

        predicts = model.predict(x_test)
        labels = y_test
        reference_tags = reference_tags_test
        evaluate_fine_tune_model(predicts, labels, reference_tags)
    else:
        predicts = model.predict(x_test)
        labels = y_test
        reference_tags = reference_tags_test
        evaluate_fine_tune_model(predicts, labels, reference_tags)

bert_indoorlocation_train_with_label()

