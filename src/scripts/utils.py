import numpy as np

from os.path import abspath
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors

def make_embedding(texts, embedding_path, max_features):
    embedding_path = abspath(embedding_path)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if embedding_path.endswith('.vec'):
        embedding_index = dict(get_coefs(*o.strip().split(" "))
                               for o in open(embedding_path))
        mean_embedding = np.mean(np.array(list(embedding_index.values())))
    elif embedding_path.endswith('bin'):
        embedding_index = KeyedVectors.load_word2vec_format(
            embedding_path, binary=True)
        mean_embedding = np.mean(embedding_index.vectors, axis=0)
    embed_size = mean_embedding.shape[0]
    word_index = sorted(list({word.lower() for sentence in texts for word in sentence}))
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    i = 1
    word_map = defaultdict(lambda: nb_words)
    for word in word_index:
        if i >= max_features:
            continue
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        else:
            embedding_matrix[i] = mean_embedding
        word_map[word] = i
        i += 1

    embedding_matrix[-1] = mean_embedding
    return embed_size, word_map, embedding_matrix
