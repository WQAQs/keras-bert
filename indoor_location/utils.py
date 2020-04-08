import pandas as pd
import numpy as np
from keras_bert.backend import keras
from keras_bert import get_custom_objects, get_base_dict

TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking

def load_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file)
    dataset = dataset[:6000]  ###  ！！！！！！！！！！！ 调整一个batch使用的样本数量！！！！！！！！！！！！！
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
    sentences, labels, plus_infos = load_dataset(dataset_file)
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
    return x_train, y_train

# file_name = ".\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"
# res = gen_bert_data(file_name, seqence_len=26)
# res