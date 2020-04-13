import pandas as pd
import numpy as np
from keras_bert.backend import keras
from keras_bert import get_custom_objects, get_base_dict
import matplotlib.pyplot as plt
import math

TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking

evaluate_prediction_file = ".\\logs\\evaluate_prediction.csv"
error_distance_over1m_file = ".\\logs\\error_distance_over1m.csv"
savefig_error_distribution_file = ".\\logs\\savefig_error_distribution.png"
savefig_coordinates_distribution_file = ".\\logs\\savefig_coordinates_distribution.png"
savefig_error_cdf_file = ".\\logs\\savefig_error_cdf.png"

def calculate_distance(result):
    pred_coordinatesx,pred_coordinatesy = result[0], result[1]
    true_coordinatesx,true_coordinatesy = result[2], result[3]
    error_x2, error_y2 = math.pow(pred_coordinatesx - true_coordinatesx, 2), \
                         math.pow(pred_coordinatesy - true_coordinatesy, 2)
    error_distance = math.sqrt(error_x2 + error_y2)  # 求平方根
    return error_distance

def evaluate_fine_tune_model(predicts_location, labels_location, reference_tags):
    results = np.hstack((predicts_location, labels_location))
    results_df = pd.DataFrame(results, columns=['pred_coordinates_x', 'pred_coordinates_y',
                                                'true_coordinates_x', 'true_coordinates_y'])

    error_distance = list(map(calculate_distance, results))
    error_distance = np.array(error_distance)
    error_distance_bypoint = np.hstack((error_distance.reshape(-1, 1), reference_tags.reshape(-1, 1)))
    error_distance_bypoint_df = pd.DataFrame(error_distance_bypoint, columns=['error_distance', 'point_reference_tag'])

    evaluate_df = pd.concat([results_df, error_distance_bypoint_df], axis=1)
    evaluate_df.to_csv(evaluate_prediction_file)   # 保存evaluate_df到csv文件

    error_over1m_df = error_distance_bypoint_df[error_distance_bypoint_df['error_distance'] > 1.0]
    error_over1m_df = error_over1m_df.sort_values(by='error_distance')
    error_mean = error_over1m_df['error_distance'].mean()
    error_over1m_df.to_csv(error_distance_over1m_file, index=False, encoding='utf-8')   # 保存error_over1m_df到csv文件

    # 绘制error_distribution图
    plt.figure()
    error_len = len(error_distance)
    plt.scatter(list(range(0, error_len)), error_distance, s=6)
    plt.title("Prediction Distance Error By Point")
    plt.ylabel("Prediction Distance Error(/m)")
    plt.savefig(savefig_error_distribution_file)
    # plt.show()

    groupby_df = evaluate_df.groupby(['point_reference_tag'])
    true_coordinates_x, true_coordinates_y = evaluate_df['true_coordinates_x'].values.tolist(), evaluate_df['true_coordinates_y'].values.tolist()

    # 绘制coordinates_distribution图
    plt.figure()
    for point_reference_tag, group_data in groupby_df:
        pred_coordinates_x, pred_coordinates_y = group_data['pred_coordinates_x'].values.tolist(), group_data['pred_coordinates_y'].values.tolist()
        plt.scatter(pred_coordinates_x, pred_coordinates_y, s=6)
    plt.scatter(true_coordinates_x, true_coordinates_y, s=18, marker='p')
    plt.xlabel('coordinate_x(/m)')
    plt.ylabel('coordinate_y(/m)')
    plt.axis('equal')
    plt.axis('square')
    plt.savefig(savefig_coordinates_distribution_file)
    # plt.show()  # 在pycharm中显示绘图的窗口

    # 绘制error cdf图
    cdf(error_distance)


def cdf(data):
    # pd.DataFrame(data).to_csv("erro_cdf_dist.csv", header=None, index=False)
    hist, bins = np.histogram(data, 100)
    bins = bins[1:]
    for i in range(1, len(hist)):
        hist[i] = hist[i]+hist[i-1]
    hist = hist/len(data)
    plt.figure()
    # 设置坐标轴刻度
    my_x_ticks = np.arange(0, 10, 1)
    my_y_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.title("Prediction Distance Error CDF")
    plt.xlabel('Prediction Distance Error (/m)')
    plt.ylabel('CDF')
    plt.plot(bins, hist)
    plt.savefig(savefig_error_cdf_file)
    # plt.show()

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

def get_sentence_pairs(dataset_file):
    data_inputs, _, _ = load_dataset(dataset_file)
    sentence_pairs = []
    one_sentence_pair = []
    for i in range(len(data_inputs)):
        one_sentence = data_inputs[i]
        if i != 0 and i % 2 == 0:
            sentence_pairs.append(one_sentence_pair)
            one_sentence_pair = []
        one_sentence_pair.append(one_sentence)
    return sentence_pairs

def gen_bert_data(dataset_file,seqence_len):
    # 准备训练集数据和验证集数据
    sentences, labels, reference_tags = load_dataset(dataset_file)
    token_dict = get_base_dict()
    for s in sentences:
        for token in s:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    def gen_batch_inputs(sentences,
                         labels,
                         token_dict,
                         seq_len=512,
                         ):
        """Generate a batch of inputs and outputs for training.

        :param sentences: A list of pairs containing lists of tokens.
        :param token_dict: The dictionary containing special tokens.
        :param token_list: A list containing all tokens.
        :param seq_len: Length of the sequence.
        :param mask_rate: The rate of choosing a token for prediction.
        :param mask_mask_rate: The rate of replacing the token to `TOKEN_MASK`.
        :param mask_random_rate: The rate of replacing the token to a random word.
        :param swap_sentence_rate: The rate of swapping the second sentences.
        :param force_mask: At least one position will be masked.
        :return: All the inputs and outputs.
        """
        batch_size = len(sentences)
        unknown_index = token_dict[TOKEN_UNK]
        token_inputs, segment_inputs = [], []
        for i in range(batch_size):
            first = sentences[i]
            segment_inputs.append(([0] * (len(first) + 2))[:seq_len])
            tokens = [TOKEN_CLS] + first + [TOKEN_SEP]
            tokens = tokens[:seq_len]
            tokens += [TOKEN_PAD] * (seq_len - len(tokens))
            token_input, masked_input, mlm_output = [], [], []
            for token in tokens:
                token_input.append(token_dict.get(token, unknown_index))
            token_inputs.append(token_input)
        inputs = [np.asarray(x) for x in [token_inputs, segment_inputs]]
        outputs = labels
        # outputs = [np.asarray(np.expand_dims(x, axis=-1)) for x in [mlm_outputs, nsp_outputs]]
        return inputs, outputs
    x_train, y_train = gen_batch_inputs(
                    sentences,
                    labels,
                    token_dict,
                    seq_len=seqence_len
                )
    return x_train, y_train, reference_tags

# file_name = ".\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
# res = gen_bert_data(file_name, seqence_len=26)
# res