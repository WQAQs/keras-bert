import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_bert
from indoor_location import hyper_parameters as hp
from indoor_location import utils
from indoor_location import globalConfig

# from keras_bert.optimizers import AdamWarmup
# from sklearn.preprocessing import LabelBinarizer
# from keras.applications.vgg16 import VGG16
# from keras.callbacks import ModelCheckpoint, TensorBoard
# from keras.optimizers import SGD
# from keras.datasets import cifar10
# from tensorflow.python.keras.callbacks import ModelCheckpoint

# valid_ibeacon_num = 26 #有效的ap数量
# # seqence_len = valid_ibeacon_num*2+3   # 因为tokens = [TOKEN_CLS] + first + [TOKEN_SEP] + second + [TOKEN_SEP]
# seqence_len = valid_ibeacon_num + 1  # 因为tokens = [TOKEN_CLS] + first
pretrain_train_datafile_path = ".\\data\\sampleset_data\\train_dataset1.csv"
pretrain_valid_datafile_path = ".\\data\\sampleset_data\\valid_dataset1.csv"


trained_model_index = "2"
pretrained_model_index = "5"
pretrained_model_path = ".\\logs\\pretrained_bert" + pretrained_model_index + ".h5"

# pretrained_path = ".\\logs"
# config_path = os.path.join(pretrained_path, 'mybert_config.json')
# checkpoint_path = os.path.join(pretrained_path, 'mybert_model.ckpt')
# checkpoint_dir = os.path.dirname(checkpoint_path)

# rssi_token_dict, rssi_id_dict = utils.rssi_token_dict, utils.rssi_id_dict
# ap_token_dict, ap_id_dict = utils.ap_token_dict,utils.ap_id_dict

word_id_map_file_path = globalConfig.word_id_map_file_path

flag_retrain = True
EPOCHS = 1000
LR = 0.05
decay_steps = 30000
warmup_steps = 10000
weight_decay = 1e-3


def bert_indoorlocation_pretrain():
    # x_train, y_train, _, _ = utils.get_data_from_sentence_pairs_for_pretrain(pretrain_train_datafile_path)
    # x_valid, y_valid, _, _ = utils.get_data_from_sentence_pairs_for_pretrain(pretrain_valid_datafile_path)
    ## 准备数据
    x_train, y_train = utils.get_strimed_data_from_sentences_for_pretrain(pretrain_train_datafile_path,
                                                                          seq_len=hp.seq_len)
    x_valid, y_valid = utils.get_strimed_data_from_sentences_for_pretrain(pretrain_valid_datafile_path,
                                                                          seq_len=hp.seq_len)
    word2id_dict, id2word_dict = utils.get_word_id_map(word_id_map_file_path)
    ## 设置GPU
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        # saver = tf.train.Saver()  # 保存模型参数的saver
        print("compiling model .....")
        model = keras_bert.my_get_model(
            token_num=len(word2id_dict),
            head_num=2,
            transformer_num=4,
            embed_dim=16,
            feed_forward_dim=100,
            seq_len=hp.seq_len,
            pos_num=hp.seq_len,
            dropout_rate=0.05,
            attention_activation='gelu',
        )
        if flag_retrain:
            model.load_weights(pretrained_model_path)

        # 初始化模型和参数
        keras_bert.compile_model(
            model,
            learning_rate=LR,
            decay_steps=decay_steps,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
        )
        model.summary()
            # data = model.to_json()
            #
            # # 保存模型到json文件
            # with open('./bert_model_config.json', 'w') as file:
            #     file.write(data)

            # Create a callback that saves the model's weights
            # cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
            #                                               save_weights_only=True,
            #                                               verbose=1)
            # cp_callback = ModelCheckpoint(filepath=checkpoint_path,
            #                                               save_weights_only=True,
            #                                               verbose=1)
            #
            # early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=30)
            # callback_lists = [early_stop]

            # # 创建一个权重文件保存文件夹logs
            # log_dir = "logs/"
            # # 记录所有训练过程，每隔一定步数记录最大值
            # tensorboard = TensorBoard(log_dir=log_dir)
            # checkpoint = ModelCheckpoint(log_dir + "bert_model.h5",##!!!这里只能是h5格式
            #                              monitor="val_loss",
            #                              mode='min',
            #                              save_weights_only=True,
            #                              save_best_only=True,
            #                              verbose=1,
            #                              period=1)
            #
            # callback_lists = [tensorboard, checkpoint]

        print("training network...")
        # Train the model with the new callback
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=5)
        # H = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
        #               batch_size=128, epochs=EPOCHS, callbacks=[early_stopping])

        # saver = tf.train.Saver()
        # saver.save(session, checkpoint_path)
        # current_path = os.path.dirname(os.path.abspath(__file__))
        # model_path = os.path.join(current_path, 'test_indoorlocation_pretrain_bert_model1.h5')

        # if os.path.exists(model_path):
        # if flag_retrain:
        #     steps_per_epoch = 1000
        #     model = keras.models.load_model(
        #         pretrained_model_path,
        #         custom_objects=get_custom_objects(),
        #     )
        # else:
        #     steps_per_epoch = 100
        #     model = get_model(
        #         token_num=len(rssi_token_dict),
        #         head_num=2,
        #         transformer_num=2,
        #         embed_dim=12,
        #         feed_forward_dim=100,
        #         seq_len=seqence_len,
        #         pos_num=seqence_len,
        #         dropout_rate=0.05,
        #         attention_activation='gelu',
        #     )
        #     compile_model(
        #         model,
        #         learning_rate=1e-3,
        #         decay_steps=30000,
        #         warmup_steps=10000,
        #         weight_decay=1e-3,
        #     )

        # def _generator():
        #     while True:
        #         yield gen_batch_inputs(
        #             sentence_pairs,
        #             rssi_token_dict,
        #             token_list,
        #             seq_len=seqence_len,
        #             mask_rate=0.3,
        #             swap_sentence_rate=1.0,
        #         )
        # model.fit_generator(
        #     generator=_generator(),
        #     steps_per_epoch=steps_per_epoch,
        #     epochs=1,
        #     validation_data=_generator(),
        #     validation_steps=steps_per_epoch // 10,
        # )
        model.save(pretrained_model_path)

        ## 分析预训练出来的模型对所有样本预测的token结果

        #
        # predicts = model.predict(x_train)
        # predicts_mlm_ids = np.argmax(predicts[0], axis=-1)
        # real_mlm_ids = list(map(lambda x: np.squeeze(x, axis=-1), y_train[0]))
        # utils.evaluate_pretrain_model(predicts_mlm_ids, real_mlm_ids)
        utils.evaluate_pretrain_model(model, x_test=x_valid, y_test=y_valid)



bert_indoorlocation_pretrain()