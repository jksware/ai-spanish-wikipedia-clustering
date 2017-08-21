#
#  Wikipedia Clustering
#  Copyright (C) 2015 Juan Carlos Pujol Mainegra, Damian Valdés Santiago
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.  
#

from __future__ import print_function
# -*- coding:utf-8 -*-
__author__ = 'Juanca'

CLUSTER_ID1 = 69
CLUSTER_ID2 = 70

import sqlite3
import json
from pprint import pprint
from time import time
from itertools import combinations
from random import sample

import nltk
from nltk.stem import *
from nltk.tokenize import word_tokenize

import numpy as np

from scipy.spatial import distance

from sklearn import metrics
from sklearn.cluster import *
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer

from openclClustering import KMeansOpenCL


spanish_sentence_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
spanish_stopwords = set(nltk.corpus.stopwords.words('spanish'))
spanish_stemmer = SnowballStemmer('spanish')


TOPICS_SELECT_SQL = """
SELECT topics.id, topics.[DESC], count( * ) AS count
FROM corpus_mini JOIN topics
    ON ( corpus_mini.topic = topics.id )
GROUP BY topics.id
HAVING count( * ) > 1000
ORDER BY count( * )  DESC;
"""

SELECT_TOPICS_COUNT_SQL_STR = """
SELECT count(*) FROM topics
"""

SELECT_N_ARTICLES_FROM_SUBJECT_K_TOPIC = """
SELECT *
FROM corpus_mini JOIN topics
    ON (corpus_mini.topic = topics.id)
WHERE topics.id = ?
LIMIT ?
"""


def pprint_vector(vector, columns=5, indent=1, word_fill=15, max_elements=None):
    _max_element = max_elements
    for i, scalar in enumerate(vector):
        if not max_elements:
            _max_element = len(vector)
        else:
            if _max_element//2 < i < len(vector) - _max_element//2 + 1:
                continue
            if i == _max_element // 2:
                scalar = '...'
            if i > len(vector) - _max_element//2:
                i -= len(vector) - _max_element

        if isinstance(scalar, float):
            int_part_len = len(str(int(scalar)))
            scalar = round(scalar, ndigits=word_fill - int_part_len - 1)

        if not isinstance(scalar, str) and not isinstance(scalar, unicode):
            scalar = str(scalar)

        is_new_line = (i % columns) == 0
        start_new_line = (i % columns) == (columns - 1)
        end = ('\n' if start_new_line else '\t') if i != _max_element - 1 else '\t]\n'
        to_fill = ' ' * (word_fill - len(scalar))
        start = ('\t' * indent if is_new_line else '') if i != 0 else '[\t'
        print(start + scalar + to_fill, end=end)


def pprint_matrix(matrix, columns=5, indent=1, word_fill=15, max_elements=None, max_vectors=None):
    for i, vector in enumerate(matrix):
        if max_vectors:
            if max_vectors//2 < i < len(matrix) - max_vectors//2 + 1:
                continue
            if i == max_vectors // 2:
                vector = ['...', ] * len(vector)
        pprint_vector(vector, columns, indent, word_fill, max_elements)
    print()


def vectorize(miniwiki_db_path=None, article_count=1000, tokenize_hard=False, fixed=False, max_combs_per_round=5,
              max_features=1000):
    begin = time()

    if not miniwiki_db_path:
        miniwiki_db_path = 'miniki.db'

    connection = sqlite3.connect(miniwiki_db_path)
    connection.enable_load_extension(True)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    if fixed:
        print("Topics are a fixed list.")

        all_selectable_topics = [{'id': 69, 'desc': 'Ficha de ciclista', 'count': 1219},
                                 {'id': 70, 'desc': 'Ficha de científico', 'count': 2438}]
    else:
        all_selectable_topics = cursor.execute(TOPICS_SELECT_SQL).fetchall()

    print()
    print('Count \t\t Topic')
    print('-' * 30)
    for topic in all_selectable_topics:
        print(topic['count'], '\t\t', topic['desc'])

    topics_len = cursor.execute(SELECT_TOPICS_COUNT_SQL_STR).fetchone()[0]
    topic_count = [0, ] * topics_len

    cursor.close()

    for n_clusters in range(2, min(5, len(all_selectable_topics) + 1)):
        if not fixed:
            print("=" * 80)
            print("Size {} combinations.".format(n_clusters))

        CLUSTER_METHODS = [
            KMeans(n_clusters=n_clusters, init='random', verbose=0),
            KMeansOpenCL(n_clusters=n_clusters, max_iters=300, tol=1e-4, verbose=False),
            DBSCAN(eps=0.7, min_samples=5),
            MeanShift()
        ]
        i_combinations = list(combinations(all_selectable_topics, n_clusters))
        if not fixed:
            # from math import factorial
            # combs = lambda: factorial(len(all_selectable_topics)) // (factorial(len(all_selectable_topics) - n_clusters) * factorial(n_clusters))
            # i_combinations.__setattr__('len', combs)
            i_combinations = sample(i_combinations, max_combs_per_round)

        print("Will be trying the following sets of vector documents (id's only):")
        for j, combination in enumerate(i_combinations):
            print(str(j) + '. ', end='')
            for cluster in combination:
                print(cluster['id'], end=' ')
            print()

        for j, combination in enumerate(i_combinations):
            cluster_ids = [cluster['id'] for cluster in combination]

            print()
            print("/" * 80)
            print("Trying the following sets of vector documents:")
            for cluster in combination:
                print('\t', cluster['id'], cluster['desc'])

            elapsed = time()
            print()
            print("Reading from database...")

            cursor = connection.cursor()

            all_clusters = []
            for k, cluster in enumerate(combination):
                print('\t', cluster['id'], '\t', cluster['desc'], end='\t')

                cluster_id = cluster['id']
                cursor.execute(SELECT_N_ARTICLES_FROM_SUBJECT_K_TOPIC, (cluster_id, article_count))
                cluster = cursor.fetchall()

                print("Read {} articles".format(len(cluster)))

                all_clusters.extend(cluster)

            cursor.close()

            print("Elapsed time (s):", time() - elapsed)
            elapsed = time()

            print()
            print("Performing tokenization, stopwords removal and stemming...")

            docs = []
            document_topics = []

            for l, row in enumerate(all_clusters):
                if tokenize_hard:
                    token_doc = row['text']
                    token_doc = spanish_sentence_tokenizer.tokenize(token_doc)
                    token_doc = [word_tokenize(sentence) for sentence in token_doc]
                    token_doc = [[spanish_stemmer.stem(word) for word in sentence
                                  if word not in spanish_stopwords and len(word) > 2]
                                 for sentence in token_doc]

                    # words = set()
                    # for sentence in token_doc:
                    #     words.update(sentence)
                    # print(len(words))

                    doc = ''
                    for sentence in token_doc:
                        for word in sentence:
                            doc += word + ' '

                title = row['title']
                doc = row['text']
                topic_desc = row['desc']
                topic_id = row['topic']

                topic_count[topic_id] += 1

                document_topics.append([topic_id, topic_desc])
                docs.append(doc)

            print("Elapsed time (s):", time() - elapsed)
            elapsed = time()

            print()
            print("Vectorizing...")

            vectorizer = TfidfVectorizer(min_df=1, max_features=max_features, stop_words=spanish_stopwords)
            # vector_space = vectorizer.fit_transform(docs)
            # features = vectorizer.get_feature_names()
            # vector_space = vector_space.toarray()

            all_features = set()
            features_from_cluster = []
            testing_set = []
            testing_set_topics = []

            for i in range(n_clusters):
                train_set = docs[i * article_count:i * article_count + article_count // 2]
                testing_set.extend(docs[i * article_count + article_count//2: (i + 1) * article_count])
                testing_set_topics.extend(document_topics[i * article_count + article_count//2: (i + 1) * article_count])
                vectorizer.fit_transform(train_set)
                train_set_features = set(vectorizer.get_feature_names())

                all_features.update(train_set_features)
                features_from_cluster.append(train_set_features)

            # delete_features = set()
            # for i0 in range(n_clusters):
            #     for i1 in range(i, n_clusters):
            #         common_features = features_from_cluster[i0].symmetric_difference(features_from_cluster[i1])
            #         delete_features.update(common_features)
            #
            # trained_features = all_features.difference(delete_features)
            trained_features = all_features

            docs = testing_set
            document_topics = testing_set_topics

            vectorizer = TfidfVectorizer(min_df=1, max_features=max_features, stop_words=spanish_stopwords,
                                         vocabulary=list(trained_features))
            vector_space = vectorizer.fit_transform(docs)
            vector_space = vector_space.toarray()

            features = vectorizer.get_feature_names()

            print("Elapsed time (s):", time() - elapsed)

            # print()
            # print("Principal component analysis...")
            # elapsed = time()

            # pca = PCA(n_components=100)
            # pca.fit(vector_space)
            # vector_space = pca.components_
            # print('Vector space ', len(vector_space))

            # print("Elapsed time (s):", time() - elapsed)

            print()
            print("Vector space is on the reals with dimensions({} x {}) and vectorizer {}:".format(
                len(vector_space), len(vector_space[0]), vectorizer.__class__.__name__))
            print('-' * 10)
            pprint_matrix(vector_space, columns=10, indent=1, word_fill=5, max_elements=10, max_vectors=20)

            print("Features are:")
            print('-' * 10)
            pprint_vector(features, columns=5, indent=1)
            print()

            print("For all the algorithms runs of the same sets, the label number does not matter.\n"
                  "What matters is that if there is a 1-to-1 relation from the assigned labels set to the \n"
                  "correct labels set. Keeping that in mind, the correct labels are:")
            print('-' * 10)
            correct_labels = np.array([vector[0] for vector in document_topics])
            pprint_vector(correct_labels, columns=20, indent=1, word_fill=3, max_elements=100)
            print()

            # if fixed:
            if False:
                print()
                print("Writing JSON out...")
                with open('vector_space.json', 'w') as vector_space_file:
                    json.dump([[scalar for scalar in vector] for vector in vector_space], vector_space_file)
                    vector_space_file.flush()

                with open('document_topics.json', 'w') as document_topics_file:
                    json.dump(document_topics, document_topics_file)
                    document_topics_file.flush()

            print()
            print("Converting vector space to float32 (instead of Python's custom float).\n"
                  "This step is key to the OpenCL KMeans.")
            vector_space = vector_space.astype(np.float32)

            print("Vector space size is")
            print("{} bytes, which is about {} kbytes, which is about {} mbytes.".format(
                  vector_space.nbytes,
                  vector_space.nbytes // 1024,
                  vector_space.nbytes // 1024 // 1024
            ))

            print("Trying various methods...")

            for l, cluster_method in enumerate(CLUSTER_METHODS):
                print("*" * 80)
                print("Cluster method {}, algorithm number {} out of {}".format(cluster_method.__class__.__name__,
                                                                                l + 1, len(CLUSTER_METHODS)))

                elapsed = time()
                if isinstance(cluster_method, MeanShift):
                    bandwidth = estimate_bandwidth(vector_space, quantile=0.55, n_samples=5)
                    cluster_method = MeanShift(bandwidth=bandwidth, bin_seeding=True)

                cluster_method.fit(vector_space)

                if isinstance(cluster_method, KMeansOpenCL):
                    cluster_method.kmeans(initial_clusters=None)

                print("Elapsed time (s):", time() - elapsed)
                elapsed = time()

                assigned_labels = cluster_method.labels_.astype(int)

                if not isinstance(cluster_method, DBSCAN):
                    cluster_ids = set(assigned_labels)
                    for cluster_id in cluster_ids:
                        cluster_count = sum(1 if label == cluster_id else 0 for label in assigned_labels )
                        print("Cluster {} assignments where {}.".format(cluster_id, cluster_count))

                print()
                print("Labels assigned by the clustering algorithm are:")
                pprint_vector(assigned_labels, columns=20, indent=1, word_fill=3, max_elements=100)
                print()

                print()
                print("Cluster recognized by the clustering algorithm")
                print(len(set(assigned_labels)))
                print()

                print("Some stats")
                print('-' * 10)
                print()

                # -1 to 1, 1
                print("\tRand Index")
                print('\t' + '-' * 10)
                print('\t',  metrics.adjusted_rand_score(correct_labels, assigned_labels))
                print()

                # 0 to 1, 1
                print("\tMutual Information")
                print('\t' + '-' * 10)
                print('\t',  metrics.adjusted_mutual_info_score(correct_labels, assigned_labels))
                print()

                h_c_v_measure = metrics.homogeneity_completeness_v_measure(correct_labels, assigned_labels)

                # 0 to 1, 1
                print("\tHomogeneity")
                print('\t' + '-' * 10)
                print('\t',  h_c_v_measure[0])
                print()

                # 0 to 1, 1
                print("\tCompleteness")
                print('\t' + '-' * 10)
                print('\t',  h_c_v_measure[1])
                print()

                # 0 to 1, 1
                print("\tV-Measure")
                print('\t' + '-' * 10)
                print('\t',  h_c_v_measure[2])
                print()

                try:
                    # -1 to 1, 1.
                    D = distance.squareform(distance.pdist(vector_space))
                    score = metrics.silhouette_score(D, assigned_labels, metric='euclidean')
                    print("\tSilhouette Coefficient")
                    print('\t' + '-' * 10)
                    print('\t', score)
                    print()
                except:
                    pass

                print("Elapsed time (s):", time() - elapsed)

    elapsed = time() - begin

    print("The whole test ran in a total of {} seconds, which is about {} minutes.".format(elapsed, elapsed // 60))
    print("That's all, folks.")


vectorize(miniwiki_db_path='c:\wiki\miniwiki.db', tokenize_hard=False, fixed=False, max_features=1000)
