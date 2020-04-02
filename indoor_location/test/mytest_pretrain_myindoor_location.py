import os
import numpy as np
import tensorflow as tf
from keras_bert.backend import keras
from keras_bert.backend import backend as K
from keras_bert import (get_model, compile_model, get_base_dict, gen_batch_inputs, get_token_embedding,
                        get_custom_objects, set_custom_objects, mytest_get_model,load_trained_model_from_checkpoint)
from indoor_location.utils import get_sentence_pairs
seqence_len = 26  #有效的ap数量
pretrain_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
train_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
flag_retrain = False
MODEL_DIR = "..\\indoor_location\\model\\"
pretrained_model_path = MODEL_DIR + "pretrained_bert1.h5"

def bert_indoorlocation_fit():
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # saver = tf.train.Saver(max_to_keep=3)

        current_path = os.path.dirname(os.path.abspath(__file__))
        # model_path = os.path.join(current_path, 'test_indoorlocation_pretrain_bert_model1.h5')
        # sentence_pairs = [
        #     [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
        #     [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
        #     [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
        #     [['you', 'are', 'beautiful'], ['yes', 'I', 'LOVE', 'you']]
        # ]
        sentence_pairs = get_sentence_pairs(pretrain_datafile_name)
        token_dict = get_base_dict()
        for pairs in sentence_pairs:
            for token in pairs[0] + pairs[1]:
                if token not in token_dict:
                    token_dict[token] = len(token_dict)
        token_list = list(token_dict.keys())
        # if os.path.exists(model_path):
        if flag_retrain:
            steps_per_epoch = 1000
            model = keras.models.load_model(
                pretrained_model_path,
                custom_objects=get_custom_objects(),
            )
        else:
            steps_per_epoch = 1000
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
            compile_model(
                model,
                learning_rate=1e-3,
                decay_steps=30000,
                warmup_steps=10000,
                weight_decay=1e-3,
            )
        model.summary()
        def _generator():
            while True:
                yield gen_batch_inputs(
                    sentence_pairs,
                    token_dict,
                    token_list,
                    seq_len=seqence_len,
                    mask_rate=0.3,
                    swap_sentence_rate=1.0,
                )
        model.fit_generator(
            generator=_generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=1,
            validation_data=_generator(),
            validation_steps=steps_per_epoch // 10,
        )
        model.save(pretrained_model_path)

        for inputs, outputs in _generator():
            predicts = model.predict(inputs)
            outputs = list(map(lambda x: np.squeeze(x, axis=-1), outputs))
            predicts = list(map(lambda x: np.argmax(x, axis=-1), predicts))
            batch_size, seq_len = inputs[-1].shape
            for i in range(batch_size):
                match, total = 0, 0
                for j in range(seq_len):
                    if inputs[-1][i][j]:
                        total += 1
                        if outputs[0][i][j] == predicts[0][i][j]:
                            match += 1
                # self.assertGreater(match, total * 0.9)
                res1 = match >= total * 0.9

            # self.assertTrue(np.allclose(outputs[1], predicts[1]))
            res2 = np.allclose(outputs[1], predicts[1])

            break

bert_indoorlocation_fit()