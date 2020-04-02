import os
import numpy as np
import tensorflow as tf
from keras_bert.backend import keras
from keras_bert.backend import backend as K
from keras_bert import (get_model, compile_model, get_base_dict, gen_batch_inputs, get_token_embedding,
                        get_custom_objects, set_custom_objects, mytest_get_model,loader,my_gen_batch_inputs)
from indoor_location.utils import (get_sentence_pairs,h5_model_file2json_model_file,load_dataset)
from keras_bert.layers import get_inputs

seqence_len = 26  #有效的ap数量
pretrain_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
train_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
flag_retrain = True
MODEL_DIR = "..\\indoor_location\\model\\"
pretrained_model_path = MODEL_DIR + "pretrained_bert1.h5"
trained_model_path = MODEL_DIR + "trained_bert1.h5"

flag_retrain = True
LR = 0.001
EPOCHS = 10
BATCH_SIZE = 128

def indoorlocation_train_bert_with_label():
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        current_path = os.path.dirname(os.path.abspath(__file__))
        h5model_path = os.path.join(current_path, 'test_indoorlocation_pretrain_bert_model.h5')
        model_path = os.path.join(current_path, 'test_indoorlocation_train_bert_model.h5')

        sentence_pairs = get_sentence_pairs(pretrain_datafile_name)
        token_dict = get_base_dict()
        for pairs in sentence_pairs:
            for token in pairs[0] + pairs[1]:
                if token not in token_dict:
                    token_dict[token] = len(token_dict)
        token_list = list(token_dict.keys())

        if os.path.exists(h5model_path):
            current_path = os.path.dirname(os.path.abspath(__file__))
            model = keras.models.load_model(
                h5model_path,
                custom_objects=get_custom_objects(),
            )
            # jsonmodel_path = os.path.join(current_path, 'test_indoorlocation_pretrain_bert_model.json')
            # h5_model_file2json_model_file(h5model_path, jsonmodel_path)
            # model, config = loader.build_model_from_config(
            #     config_file=jsonmodel_path,
            #     training=False,
            #     trainable=True,
            #     )
            # inputs = get_inputs(seq_len=seqence_len)

            # inputs, extract = mytest_get_model(token_num=len(token_dict),training=False)
            # # extract = model.get_layer('Extract').output
            # outputs = keras.layers.Dense(units=2, activation='relu')(extract)

            inputs, transformed = mytest_get_model(token_num=len(token_dict), training=False)
            optimizer = keras.optimizers.RMSprop(LR)
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mse'],
            )
        model.summary()

        train_x, train_y, train_reference_tags = load_dataset(train_datafile_name)
        inputs, outputs = my_gen_batch_inputs(train_x,train_y)

        # model.fit(
        #     train_x,
        #     train_y,
        #     epochs=EPOCHS,
        #     batch_size=BATCH_SIZE,
        # )
        # model.save(model_path)

        predicts = model.predict(train_x)
        train_y = list(map(lambda x: np.squeeze(x, axis=-1), train_y))
        predicts = list(map(lambda x: np.argmax(x, axis=-1), predicts))
        predicts

indoorlocation_train_bert_with_label()

