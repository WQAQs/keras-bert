import os
import numpy as np
import tensorflow as tf
from keras_bert.backend import keras
from keras_bert.backend import backend as K
from keras_bert import (get_model, compile_model, get_base_dict, gen_batch_inputs, get_token_embedding,
                        get_custom_objects, set_custom_objects,load_trained_model_from_checkpoint,build_model_from_config)
from indoor_location.utils import get_sentence_pairs
from keras_bert.optimizers import AdamWarmup
from sklearn.preprocessing import LabelBinarizer
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
from keras.datasets import cifar10
from tensorflow.python.keras.callbacks import ModelCheckpoint

seqence_len = 26  #有效的ap数量
pretrain_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
train_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
flag_retrain = False
MODEL_DIR = "..\\model\\"
pretrained_model_path = MODEL_DIR + "pretrained_bert1.h5"

pretrained_path = ".\\logs"
config_path = os.path.join(pretrained_path, 'mybert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'mybert_model.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)

epochs = 2


def bert_indoorlocation_fit():
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
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:

        # saver = tf.train.Saver()  # 保存模型参数的saver

        print("compiling model .....")
        model = get_model(
            token_num=len(token_dict),
            head_num=2,
            transformer_num=2,
            embed_dim=12,
            feed_forward_dim=100,
            seq_len=seqence_len,
            pos_num=seqence_len,
            dropout_rate=0.05,
            attention_activation='gelu',
        )

        # model, _ = build_model_from_config(
        #     config_path,
        #     training=True,
        #     trainable=None)
        # 初始化模型和参数
        compile_model(
            model,
            learning_rate=1e-3,
            decay_steps=30000,
            warmup_steps=10000,
            weight_decay=1e-3,
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
        H = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                      batch_size=32, epochs=1, verbose=2)

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
        #         token_num=len(token_dict),
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
        #             token_dict,
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




        # for (x, y) in zip(x_train,y_train):
        #     predicts = model.predict(x)
        #     y = list(map(lambda x: np.squeeze(x, axis=-1), y))
        #     predicts = list(map(lambda x: np.argmax(x, axis=-1), predicts))
        #     batch_size, seq_len = x[-1].shape
        #     for i in range(batch_size):
        #         match, total = 0, 0
        #         for j in range(seq_len):
        #             if x[-1][i][j]:
        #                 total += 1
        #                 if y[0][i][j] == predicts[0][i][j]:
        #                     match += 1
        #         # self.assertGreater(match, total * 0.9)
        #         res1 = match >= total * 0.9
        #
        #     # self.assertTrue(np.allclose(y[1], predicts[1]))
        #     res2 = np.allclose(y[1], predicts[1])
        #
        #     break

bert_indoorlocation_fit()