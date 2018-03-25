# -*- coding: utf-8 -*-
import numpy as np


def make_glovevec(glovepath, max_features, embed_size, word_index, veclen=300):
    embeddings_index = {}
    f = open(glovepath)
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs.reshape(-1)
    f.close()

    #all_embs = np.stack(embeddings_index.values())
    #emb_mean, emb_std = all_embs.mean(), all_embs.std()
    nb_words = min(max_features, len(word_index))
    #embedding_matrix = np.random.normal(emb_mean, emb_std,
                                        #(nb_words, embed_size))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
