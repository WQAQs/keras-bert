import pandas as pd
import numpy as np
from keras_bert.backend import keras
from keras_bert import get_custom_objects

def h5_model_file2json_model_file(h5model_path,jsonmodel_path):
    model = keras.models.load_model(
        h5model_path,
        custom_objects=get_custom_objects(),
    )
    model_json = model.to_json()
    with open(jsonmodel_path, 'w') as file:
        file.write(model_json)

def load_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file)
    reference_tags = dataset.values[:, 0]
    # data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
    coordinates = dataset.values[:, 1:3] #包括index=1，不包括index=3
    cluster_tags = dataset.values[:, 4]
    data_inputs = dataset.values[:, 5:]  # 包括index=5
    data_inputs = data_inputs.astype(np.int32)
    data_inputs = data_inputs.astype(np.str)
    data_inputs = data_inputs.tolist()
    res = data_inputs[:100]  ###  ！！！！！！！！！！！ 调整一个batch使用的样本数量！！！！！！！！！！！！！
    return res, coordinates, reference_tags

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

# file_name = "..\\data\\sampleset_data\\sampleset_day4_points15_average_interval_500ms.csv"
# res = get_sentence_pairs(file_name)
# res