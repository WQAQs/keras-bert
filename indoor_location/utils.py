import pandas as pd
import numpy as np
from keras_bert.backend import keras
from keras_bert import get_base_dict
import matplotlib.pyplot as plt
import math
import os
import keras_bert
from indoor_location import hyper_parameters as hp
from indoor_location import globalConfig
import ast
from sklearn.utils import shuffle

TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking

base_tokens = [TOKEN_MASK]

token_id_from_numerical_order_file_path = ".\\logs\\token_id_from_numerical_order.csv"
token_id_from_dataset_order_file_path = ".\\logs\\token_id_from_dataset__order2.csv"

ap_map_file_path = "..\\data\\sampleset_data\\ap_map.csv"
rssi_map_file_path = "..\\data\\sampleset_data\\rssi_map.csv"

all_mask_real_tokens_file_path = ".\\logs\\all_mask_real_tokens.csv"
all_mask_predict_tokens_file_path = ".\\logs\\all_mask_predict_tokens.csv"
all_match_res_file_path = ".\\logs\\all_match_res.csv"
all_predicts_mlm_tokens_file_path = ".\\logs\\all_predicts_mlm_tokens.csv"
all_real_mlm_tokens_file_path = ".\\logs\\all_real_mlm_tokens.csv"

evaluate_prediction_file_name = "evaluate_prediction.csv"
error_distance_over1m_file_name = "error_distance_over1m.csv"
savefig_error_distribution_file_name = "savefig_error_distribution.png"
savefig_coordinates_distribution_file_name = "savefig_coordinates_distribution.png"
savefig_error_cdf_file_name = "savefig_error_cdf.png"

rssi_token_dict, rssi_id_dict = {}, {}
ap_token_dict, ap_id_dict = {}, {}

def gen_label_rate_dataset_file(origin_file, label_rate, target_file):
    df = pd.read_csv(origin_file)
    shuffle(df)
    df.sample(frac=label_rate, random_state=0).to_csv(target_file)


def load_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file)
    # dataset = dataset[:6000]  ###  ！！！！！！！！！！！ 调整一个batch使用的样本数量！！！！！！！！！！！！！
    reference_tags = dataset.values[:, 0]
    # data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
    coordinates = dataset.values[:, 1:3] #包括index=1，不包括index=3
    cluster_tags = dataset.values[:, 4]
    data_inputs = dataset.values[:, 5:]  # 包括index=5
    data_inputs = data_inputs.astype(np.int32)
    data_inputs = data_inputs.astype(np.str)
    data_inputs = data_inputs.tolist()
    # res = data_inputs[:100]  ###  ！！！！！！！！！！！ 调整一个batch使用的样本数量！！！！！！！！！！！！！
    return data_inputs, coordinates, reference_tags

# def load_data(dataset_file):
#     inputs, outputs, reference_tags = load_dataset(dataset_file)
#
#     global tokenizer
#     indices, sentiments = [], []
#     for folder, sentiment in (('neg', 0), ('pos', 1)):
#         folder = os.path.join(path, folder)
#         for name in tqdm(os.listdir(folder)):
#             with open(os.path.join(folder, name), 'r') as reader:
#                 text = reader.read()
#             ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
#             indices.append(ids)
#             sentiments.append(sentiment)
#     items = list(zip(indices, sentiments))
#     np.random.shuffle(items)
#     indices, sentiments = zip(*items)
#     indices = np.array(indices)
#     mod = indices.shape[0] % BATCH_SIZE
#     if mod > 0:
#         indices, sentiments = indices[:-mod], sentiments[:-mod]
#     return [indices, np.zeros_like(indices)], np.array(sentiments)
## 从样本集中生成rssi_map
# def gen_rssi_map_from_dataset(sampleset_file, rssi_map_file_path):
#     # rssi_tokens = [x for x in range(-128, 0)]
#     # rssi_tokens += base_tokens
#     # rssi_ids = [i for i in range(len(rssi_tokens))]
#     # pd.DataFrame(data={"rssi_token": rssi_tokens, "rssi_id": rssi_ids}).to_csv(rssi_map_file_path)
#     # rssi_token_dict = dict(zip(rssi_tokens, rssi_ids))
#     # rssi_id_dict = dict(zip(rssi_ids, rssi_tokens))
#
#     if not os.path.exists(token_id_from_dataset_order_file_path):
#         # dataset_token_dict = {TOKEN_MASK: 0}  # dataset_token_dict：key为token，value为token对应的id
#         # id_dict = {0: TOKEN_MASK} # rssi_id_dict：key为id，value为id对应的token
#         base_tokens = [TOKEN_MASK]
#         other_tokens = []
#         sentences, _, _ = load_dataset(sampleset_file)
#         for sentence in sentences:
#             for token in sentence:
#                 if token not in other_tokens:
#                     other_tokens.append(token)
#         other_tokens = sorted(other_tokens, key=lambda x: int(x))
#         all_tokens = base_tokens + other_tokens
#         all_ids = [i for i in range(len(all_tokens))]
#         # pd.DataFrame(data={"token": all_tokens, "id": all_ids}).to_csv(token_id_from_numerical_order_file_path)
#         pd.DataFrame(data={"rssi_token": all_tokens, "id": all_ids}).to_csv(token_id_from_dataset_order_file_path)
#     df_data = pd.read_csv(token_id_from_dataset_order_file_path)
#     tokens = df_data["rssi_token"]
#     ids = df_data["id"]
#     rssi_token_dict = dict(zip(tokens, ids))
#     rssi_id_dict = dict(zip(ids, tokens))
#     return rssi_token_dict, rssi_id_dict

def gen_word_id_map_from_valid_ap(valid_ap_file, saved_file_path):
    valid_mac_list = pd.read_csv(valid_ap_file)['mac']
    word_list = []
    word2id_dict = get_base_token2id_dict()
    id2word_dict = get_base_id2token_dict()
    id = len(word2id_dict)
    for mac in valid_mac_list:
        for i in range(-100, -40):
            word = mac+'_'+str(i)
            word_list.append(word)
            word2id_dict[word] = id
            id2word_dict[id] = word
            id += 1
    all_word = list(word2id_dict.keys())
    all_id = list(word2id_dict.values())
    pd.DataFrame(data={"word": all_word, "id": all_id}).to_csv(saved_file_path)
    return word2id_dict, id2word_dict

def get_word_id_map(word_id_map_file_path):
    csv = pd.read_csv(word_id_map_file_path)
    word_list = csv['word'].values.tolist()
    id_list = csv['id'].values.tolist()
    word2id_dict = dict(zip(word_list, id_list))
    id2word_dict = dict(zip(id_list, word_list))
    return word2id_dict, id2word_dict

# def gen_ap_map(dataset_file, ap_map_file_path):
#     dataframe = pd.read_csv(dataset_file)
#     ap_tokens = pd.read_csv(dataset_file).columns.tolist()[5:]  # 列名从第5列开始是ap的mac值
#     ap_tokens += base_tokens
#     ap_ids = [x for x in range(len(ap_tokens))]
#     pd.DataFrame(data={"ap_token": ap_tokens, "ap_id": ap_ids}).to_csv(ap_map_file_path)
#     ap_token_dict = dict(zip(ap_tokens, ap_ids))
#     ap_id_dict = dict(zip(ap_ids, ap_tokens))
#     return ap_token_dict, ap_id_dict
#
# def get_id_data_from_sentence_pairs_for_pretrain(file):
#     sentence_pairs = get_sentence_pairs(file)
#     if not os.path.exists(token_id_from_numerical_order_file_path):
#         dataset_token_dict = get_base_dict()  # dataset_token_dict：key为token，value为token对应的id
#         id_dict = get_base_id2token_dict()  # rssi_id_dict：key为id，value为id对应的token
#         df_data = [[TOKEN_PAD, 0],
#                    [TOKEN_UNK, 1],
#                    [TOKEN_CLS, 2],
#                    [TOKEN_SEP, 3],
#                    [TOKEN_MASK, 4]
#         ]
#         base_tokens = [TOKEN_PAD, TOKEN_UNK, TOKEN_CLS, TOKEN_SEP, TOKEN_MASK]
#         other_tokens = []
#
#         for pairs in sentence_pairs:
#             for token in pairs[0] + pairs[1]:
#                 if token not in dataset_token_dict:
#                     n = len(dataset_token_dict)
#                     dataset_token_dict[token] = n
#                     other_tokens.append(token)
#                     id_dict[n] = token
#                     df_data.append([token, n])
#         other_tokens = sorted(other_tokens, key=lambda x: int(x))
#         all_tokens = base_tokens + other_tokens
#         all_ids = [i for i in range(len(all_tokens))]
#         pd.DataFrame(data={"token": all_tokens, "id": all_ids}).to_csv(token_id_from_numerical_order_file_path)
#         pd.DataFrame(df_data, columns=["token", "id"]).to_csv(token_id_from_dataset_order_file_path)
#     df_data = pd.read_csv(token_id_from_numerical_order_file_path)
#     tokens = df_data["token"]
#     ids = df_data["id"]
#     token_dict = dict(zip(tokens, ids))
#     id_dict = dict(zip(ids, tokens))
#
#     token_list = list(token_dict.keys())
#
#     x, y = keras_bert.gen_batch_inputs(
#         sentence_pairs,
#         token_dict,
#         token_list,
#         seq_len=hp.seq_len,
#         mask_rate=0.3,
#         swap_sentence_rate=0.5,
#     )
#     return x, y, token_dict, id_dict
#
# def get_id_data_from_sentences_for_pretrain(dataset_file,
#                                             mask_rate=0.15,
#                                             mask_mask_rate=0.8,
#                                             mask_random_rate=0.1,
#                                             force_mask=True):
#     """
#
#     :param batch_size:
#     :param rssi_list:numpy.ndarray
#     :param ap_token_dict:numpy.ndarray
#     :param ap_list:list<list>
#     :param mask_rate:
#     :param mask_mask_rate:
#     :param mask_random_rate:
#     :param force_mask:
#     :return:
#     """
#     global rssi_token_dict, rssi_id_dict,ap_token_dict, ap_id_dict
#     rssi_list, _, _ = load_dataset(dataset_file)
#     ap_list = pd.read_csv(dataset_file).columns.tolist()[5:]  # 列名从第5列开始是ap的mac值
#     rssi_token_dict, rssi_id_dict = gen_rssi_map_from_dataset(dataset_file, rssi_map_file_path)
#     ap_token_dict, ap_id_dict = gen_ap_map(dataset_file, ap_map_file_path)
#
#     base_dict = get_base_dict()
#     # ap_unknown_index = ap_token_dict[TOKEN_UNK]
#     # rssi_unknown_index = rssi_token_dict[TOKEN_UNK]
#
#     ap_inputs, rssi_inputs, masked_inputs = [], [], []
#     ap_mlm_outputs, rssi_mlm_outputs = [], []
#     has_mask = False
#     for i in range(len(rssi_list)):
#         ap_input, masked_input, rssi_input = [], [], []
#         ap_mlm_output, rssi_mlm_output = [], []
#         aps = ap_list
#         rssis = rssi_list[i]
#         # ap_mlm_output.append([]ap_token_dict.get(aps, ap_unknown_index))
#         # rssi_mlm_output.append(rssi_token_dict.get(rssis, rssi_unknown_index))
#         ap_mlm_output = [ap_token_dict.get(x) for x in aps]
#         rssi_mlm_output = [rssi_token_dict.get(x) for x in rssis]
#         for j in range(len(aps)):
#             ap_token = aps[j]
#             rssi_token = rssis[j]
#             if ap_token not in base_dict and np.random.random() < mask_rate:
#                 has_mask = True
#                 masked_input.append(1)
#                 r = np.random.random()
#                 if r < mask_mask_rate:
#                     ap_input.append(ap_token_dict[TOKEN_MASK])
#                     rssi_input.append(rssi_token_dict[TOKEN_MASK])
#                 elif r < mask_mask_rate + mask_random_rate:
#                     ap_input.append(np.random.randint(0, len(ap_token_dict)))
#                     rssi_input.append(np.random.randint(20, 90))
#                 else:
#                     ap_input.append(ap_token_dict.get(ap_token))
#                     rssi_input.append(rssi_token_dict.get(rssi_token))
#             else:
#                 masked_input.append(0)
#                 ap_input.append(ap_token_dict.get(ap_token))
#                 rssi_input.append(rssi_token_dict.get(rssi_token))
#         if force_mask and not has_mask:
#             masked_input[0] = 1
#         ap_inputs.append(ap_input)
#         rssi_inputs.append(rssi_input)
#         masked_inputs.append(masked_input)
#         ap_mlm_outputs.append(ap_mlm_output)
#         rssi_mlm_outputs.append(rssi_mlm_output)
#
#     inputs = [np.asarray(x) for x in [ap_inputs, rssi_inputs, masked_inputs]]
#     # outputs = [np.asarray(np.expand_dims(x, axis=-1)) for x in [ap_mlm_outputs, rssi_mlm_outputs]]
#     # outputs = np.asarray(np.expand_dims(x, axis=-1)) for x in rssi_mlm_outputs
#     outputs = np.asarray([np.asarray(np.expand_dims(x, axis=-1)) for x in rssi_mlm_outputs])
#     return inputs, outputs

# def get_sentence_pairs(dataset_file):
#     data_inputs, _, _ = load_dataset(dataset_file)
#     sentence_pairs = []
#     one_sentence_pair = []
#     for i in range(len(data_inputs)):
#         one_sentence = data_inputs[i]
#         if i != 0 and i % 2 == 0:
#             sentence_pairs.append(one_sentence_pair)
#             one_sentence_pair = []
#         one_sentence_pair.append(one_sentence)
#     return sentence_pairs

def strim_and_padding(list, seq_len):
    for item in list:
        item[:] = item[:seq_len]  # 多余的去掉
        n_padding = seq_len - len(item)  # 不足的补齐
        for i in range(n_padding):
            item.append(0)


# 存储在csv中的数组，在读取后变成了str类型：如 '[1782, 2004, 2841, 4101, 1781, 2001, 3055,
                                # 3889, 2841, 1781, 1998, 3888, 2839, 1998, 3055, 3888, 2841, 1782, 2422, 1997, 2842, 2639]'
                                # 要解析还原一下
def csvstr2list(data):
    new_data = []
    for i in data:
        ## ast 可以做string与list,tuple,dict之间的类型转换
        temp = ast.literal_eval(i)
        new_data.append(temp)
    return new_data

def get_strimed_data_from_sentences_for_pretrain(dataset_file,
                                            seq_len=30,
                                            mask_rate=0.15,
                                            mask_mask_rate=0.8,
                                            mask_random_rate=0.1,
                                            force_mask=True):
    """

    :param batch_size:
    :param rssi_list:numpy.ndarray
    :param ap_token_dict:numpy.ndarray
    :param ap_list:list<list>
    :param mask_rate:
    :param mask_mask_rate:
    :param mask_random_rate:
    :param force_mask:
    :return:
    """
    t =pd.read_csv(dataset_file)["mac_rssi_sentence"].values
    sentences = pd.read_csv(dataset_file)["mac_rssi_sentence"].values.tolist()
    sentence_list = csvstr2list(sentences)
    strim_and_padding(sentence_list, seq_len)
    base_id2token_dict = get_base_id2token_dict()
    base_token2id_dict = get_base_token2id_dict()
    word2id_dict_len = len(pd.read_csv(globalConfig.word_id_map_file_path)['id'].values.tolist())
    sentence_inputs, masked_inputs = [], []
    mlm_outputs = []
    has_mask = False
    for sentence in sentence_list:
        sentence_input, masked_input = [], []
        mlm_output = [token_id for token_id in sentence]
        for token_id in sentence:
            if token_id not in base_id2token_dict and np.random.random() < mask_rate:
                has_mask = True
                masked_input.append(1)
                r = np.random.random()
                if r < mask_mask_rate:
                    sentence_input.append(base_token2id_dict[TOKEN_MASK])
                elif r < mask_mask_rate + mask_random_rate:
                    sentence_input.append(np.random.randint(0, word2id_dict_len))
                else:
                    sentence_input.append(token_id)
            else:
                masked_input.append(0)
                sentence_input.append(token_id)
        if force_mask and not has_mask:
            masked_input[0] = 1
        sentence_inputs.append(sentence_input)
        masked_inputs.append(masked_input)
        mlm_outputs.append(mlm_output)
    inputs = [np.asarray(x) for x in [sentence_inputs, masked_inputs]]
    outputs = np.asarray([np.asarray(np.expand_dims(x, axis=-1)) for x in mlm_outputs])
    return inputs, outputs


def gen_fine_tune_bert_data(dataset_file,seq_len):
    # 准备训练集数据和验证集数据
    # sentences, labels, reference_tags = load_dataset(dataset_file)
    csv = pd.read_csv(dataset_file)
    sentences = csv["mac_rssi_sentence"].values.tolist()
    sentence_list = csvstr2list(sentences)
    strim_and_padding(sentence_list, seq_len)

    coordinate_x = csv["coordinate_x"].values.tolist()
    coordinate_y = csv["coordinate_y"].values.tolist()
    labels = csv[["coordinate_x", "coordinate_y"]].values # 已经是array了
    reference_tags = csv['reference_tag'].values.tolist()
    #
    # token_dict = get_base_dict()
    # for s in sentences:
    #     for token in s:
    #         if token not in token_dict:
    #             token_dict[token] = len(token_dict)
    # def gen_batch_inputs(sentences,
    #                      labels,
    #                      token_dict,
    #                      seq_len=512,
    #                      ):
    #     """Generate a batch of inputs and outputs for training.
    #
    #     :param sentences: A list of pairs containing lists of tokens.
    #     :param token_dict: The dictionary containing special tokens.
    #     :param token_list: A list containing all tokens.
    #     :param seq_len: Length of the sequence.
    #     :param mask_rate: The rate of choosing a token for prediction.
    #     :param mask_mask_rate: The rate of replacing the token to `TOKEN_MASK`.
    #     :param mask_random_rate: The rate of replacing the token to a random word.
    #     :param swap_sentence_rate: The rate of swapping the second sentences.
    #     :param force_mask: At least one position will be masked.
    #     :return: All the inputs and outputs.
    #     """
    #     batch_size = len(sentences)
    #     unknown_index = token_dict[TOKEN_UNK]
    #     token_inputs, segment_inputs = [], []
    #     for i in range(batch_size):
    #         first = sentences[i]
    #         segment_inputs.append(([0] * (len(first) + 2))[:seq_len])
    #         tokens = [TOKEN_CLS] + first + [TOKEN_SEP]
    #         tokens = tokens[:seq_len]
    #         tokens += [TOKEN_PAD] * (seq_len - len(tokens))
    #         token_input, masked_input, mlm_output = [], [], []
    #         for token in tokens:
    #             token_input.append(token_dict.get(token, unknown_index))
    #         token_inputs.append(token_input)
    #     inputs = [np.asarray(x) for x in [token_inputs, segment_inputs]]
    #     outputs = labels
    #     # outputs = [np.asarray(np.expand_dims(x, axis=-1)) for x in [mlm_outputs, nsp_outputs]]
    #     return inputs, outputs
    # x_train, y_train = gen_batch_inputs(
    #                 sentences,
    #                 labels,
    #                 token_dict,
    #                 seq_len=seq_len
    #             )
    x_train, y_train = np.asarray(sentence_list), labels
    return x_train, y_train, reference_tags

def get_base_id2token_dict():
    return {
        0: TOKEN_PAD,
        1: TOKEN_UNK,
        2: TOKEN_CLS,
        3: TOKEN_SEP,
        4: TOKEN_MASK,
    }

def get_base_token2id_dict():
    return {
        TOKEN_PAD: 0,
        TOKEN_UNK: 1,
        TOKEN_CLS: 2,
        TOKEN_SEP: 3,
        TOKEN_MASK: 4
    }

def id_list2token_list(id_list, id_dict):
    token_list = []
    for id in id_list:
        token = id_dict.get(id, TOKEN_UNK)
        token_list.append(token)
    return token_list


def all_id_lists2all_token_lists(sentences_id_data,id2word_dict):
    sentences_token_data = []
    for item in sentences_id_data:
        token_list = id_list2token_list(item, id2word_dict)
        sentences_token_data.append(token_list)
    return sentences_token_data

def evaluate_pretrain_model(model,x_test,y_test,id2word_dict):
    predicts = model.predict(x_test)
    # predicts_mlm_ids = np.argmax(predicts[0], axis=-1)
    predicts_mlm_ids = np.argmax(predicts, axis=-1)
    # real_mlm_ids = list(map(lambda x: np.squeeze(x, axis=-1), y_test[0]))
    real_mlm_ids = list(map(lambda x: np.squeeze(x, axis=-1), y_test))
    predicts_mlm_tokens = all_id_lists2all_token_lists(predicts_mlm_ids,id2word_dict)
    real_mlm_tokens = all_id_lists2all_token_lists(real_mlm_ids,id2word_dict)

    samples_num, seq_len = x_test[0].shape
    all_match_res = []
    all_mask_real_tokens = []
    all_mask_predict_tokens = []
    for i in range(samples_num):
        mask_real_tokens = []
        mask_predict_tokens = []
        match, total = 0, 0
        for j in range(seq_len):
            if x_test[-1][i][j]:
                total += 1
                mask_real_tokens.append(real_mlm_tokens[i][j])
                mask_predict_tokens.append(predicts_mlm_tokens[i][j])
                if predicts_mlm_tokens[i][j] == real_mlm_tokens[i][j]:
                    match += 1
        all_mask_real_tokens.append(mask_real_tokens)
        all_mask_predict_tokens.append(mask_predict_tokens)
        match_ratio = match / total if total else "NAN"
        temp = [total, match, match_ratio]
        all_match_res.append(temp)
    pd.DataFrame(predicts_mlm_tokens).to_csv(all_predicts_mlm_tokens_file_path, encoding='utf-8')
    pd.DataFrame(real_mlm_tokens).to_csv(all_real_mlm_tokens_file_path, encoding='utf-8')
    pd.DataFrame(all_mask_real_tokens).to_csv(all_mask_real_tokens_file_path, encoding='utf-8')
    pd.DataFrame(all_mask_predict_tokens).to_csv(all_mask_predict_tokens_file_path, encoding='utf-8')
    pd.DataFrame(all_match_res, columns=["total_mask_num", "match_num", "match_ratio"]) \
        .to_csv(all_match_res_file_path, encoding='utf-8')

def calculate_distance(result):
    pred_coordinatesx,pred_coordinatesy = result[0], result[1]
    true_coordinatesx,true_coordinatesy = result[2], result[3]
    error_x2, error_y2 = math.pow(pred_coordinatesx - true_coordinatesx, 2), \
                         math.pow(pred_coordinatesy - true_coordinatesy, 2)
    error_distance = math.sqrt(error_x2 + error_y2)  # 求平方根
    return error_distance

def calculate_distance(result):
    pred_coordinatesx,pred_coordinatesy = result[0], result[1]
    true_coordinatesx,true_coordinatesy = result[2], result[3]
    error_x2, error_y2 = math.pow(pred_coordinatesx - true_coordinatesx, 2), \
                         math.pow(pred_coordinatesy - true_coordinatesy, 2)
    error_distance = math.sqrt(error_x2 + error_y2)  # 求平方根
    return error_distance

# def evaluate_fine_tune_model(predicts_location, labels_location, reference_tags):
#     results = np.hstack((predicts_location, labels_location))
#     results_df = pd.DataFrame(results, columns=['pred_coordinates_x', 'pred_coordinates_y',
#                                                 'true_coordinates_x', 'true_coordinates_y'])
#
#     error_distance = list(map(calculate_distance, results))
#     # error_distance = np.array(error_distance)
#     # error_distance_bypoint = np.hstack((error_distance.reshape(-1, 1), reference_tags.reshape(-1, 1)))
#     # error_distance_bypoint_df = pd.DataFrame(error_distance_bypoint, columns=['error_distance', 'point_reference_tag'])
#     error_distance_bypoint_df = pd.DataFrame(data={"error_distance": error_distance,
#                                                    "point_reference_tag": reference_tags})
#
#     evaluate_df = pd.concat([results_df, error_distance_bypoint_df], axis=1)
#     evaluate_df.to_csv(evaluate_prediction_file)   # 保存evaluate_df到csv文件
#
#     error_over1m_df = error_distance_bypoint_df[error_distance_bypoint_df['error_distance'] > 1.0]
#     error_over1m_df = error_over1m_df.sort_values(by='error_distance')
#     error_mean = error_over1m_df['error_distance'].mean()
#     error_over1m_df.to_csv(error_distance_over1m_file, index=False, encoding='utf-8')   # 保存error_over1m_df到csv文件
#
#     # 绘制error_distribution图
#     plt.figure()
#     error_len = len(error_distance)
#     plt.scatter(list(range(0, error_len)), error_distance, s=6)
#     plt.title("Prediction Distance Error By Point")
#     plt.ylabel("Prediction Distance Error(/m)")
#     plt.savefig(savefig_error_distribution_file)
#     # plt.show()
#
#     groupby_df = evaluate_df.groupby(['point_reference_tag'])
#     true_coordinates_x, true_coordinates_y = evaluate_df['true_coordinates_x'].values.tolist(), evaluate_df['true_coordinates_y'].values.tolist()
#
#     # 绘制coordinates_distribution图
#     plt.figure()
#     for point_reference_tag, group_data in groupby_df:
#         pred_coordinates_x, pred_coordinates_y = group_data['pred_coordinates_x'].values.tolist(), group_data['pred_coordinates_y'].values.tolist()
#         plt.scatter(pred_coordinates_x, pred_coordinates_y, s=6)
#     plt.scatter(true_coordinates_x, true_coordinates_y, s=18, marker='p')
#     plt.xlabel('coordinate_x(/m)')
#     plt.ylabel('coordinate_y(/m)')
#     plt.axis('equal')
#     plt.axis('square')
#     plt.savefig(savefig_coordinates_distribution_file)
#     # plt.show()  # 在pycharm中显示绘图的窗口
#
#     # 绘制error cdf图
#     cdf(error_distance)


def evaluate_fine_tune_model(model, evaluate_file, results_dir):
    x_test, y_test, reference_tags_test = gen_fine_tune_bert_data(evaluate_file, hp.seq_len)
    predicts_location = model.predict(x_test)
    labels_location = y_test
    reference_tags = reference_tags_test

    results = np.hstack((predicts_location, labels_location))
    results_df = pd.DataFrame(results, columns=['pred_coordinates_x', 'pred_coordinates_y',
                                                'true_coordinates_x', 'true_coordinates_y'])

    error_distance = list(map(calculate_distance, results))
    error_distance_bypoint_df = pd.DataFrame(data={"error_distance": error_distance,
                                                   "point_reference_tag": reference_tags})

    evaluate_df = pd.concat([results_df, error_distance_bypoint_df], axis=1)
    evaluate_df.to_csv('/'.join(results_dir, evaluate_prediction_file_name))   # 保存evaluate_df到csv文件

    error_over1m_df = error_distance_bypoint_df[error_distance_bypoint_df['error_distance'] > 1.0]
    error_over1m_df = error_over1m_df.sort_values(by='error_distance')
    error_mean = error_over1m_df['error_distance'].mean()
    error_over1m_df.to_csv('/'.join(results_dir, error_distance_over1m_file_name), index=False, encoding='utf-8')   # 保存error_over1m_df到csv文件

    # 绘制error_distribution图
    plt.figure()
    error_len = len(error_distance)
    plt.scatter(list(range(0, error_len)), error_distance, s=6)
    plt.title("Prediction Distance Error By Point")
    plt.ylabel("Prediction Distance Error(/m)")
    plt.savefig('/'.join(results_dir, savefig_error_distribution_file_name))
    # plt.show()

    groupby_df = evaluate_df.groupby(['point_reference_tag'])
    true_coordinates_x, true_coordinates_y = evaluate_df['true_coordinates_x'].values.tolist(), evaluate_df['true_coordinates_y'].values.tolist()

    # 绘制coordinates_distribution图
    plt.figure()
    for point_reference_tag, group_data in groupby_df:
        pred_coordinates_x, pred_coordinates_y = group_data['pred_coordinates_x'].values.tolist(), group_data['pred_coordinates_y'].values.tolist()
        plt.scatter(pred_coordinates_x, pred_coordinates_y, s=3)
    plt.scatter(true_coordinates_x, true_coordinates_y, s=18, marker='p')
    plt.xlabel('coordinate_x(/m)')
    plt.ylabel('coordinate_y(/m)')
    plt.axis('equal')
    plt.axis('square')
    plt.savefig('/'.join(results_dir, savefig_coordinates_distribution_file_name))
    # plt.show()  # 在pycharm中显示绘图的窗口

    # 绘制error cdf图
    cdf(error_distance, results_dir)

def cdf(data, target_dir):
    # pd.DataFrame(data).to_csv("erro_cdf_dist.csv", header=None, index=False)
    hist, bins = np.histogram(data, 1000)
    bins = bins[1:]
    flag1,flag2,flag3 = True,True,True
    for i in range(1, len(hist)):
        hist[i] = hist[i]+hist[i-1]
        if flag1 and bins[i] > 1.0:
            y1 = hist[i]/len(data)
            flag1 = False
        if flag2 and bins[i] >2.0:
            y2 = hist[i]/len(data)
            flag2 = False
        if flag3 and bins[i] > 3.0:
            y3 = hist[i]/len(data)
            flag3 = False

    hist = hist/len(data)
    plt.figure()
    # 设置坐标轴刻度
    my_x_ticks = np.arange(0, 26, 1)
    my_y_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.title("Prediction Distance Error CDF")
    plt.xlabel('Prediction Distance Error (/m)')
    plt.ylabel('CDF')
    plt.plot(bins, hist)
    text = '{:.2f}<=1m  {:.2f}<=2m  {:.2f}<=3m'.format(y1, y2, y3)
    plt.text(8,0.8,text)
    plt.savefig('/'.join(target_dir, savefig_error_cdf_file_name))
    # plt.show()

# file_name = ".\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
# res = gen_bert_data(file_name, seqence_len=26)
# res