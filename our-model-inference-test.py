import time
import argparse
import os

from collections import defaultdict

from mmnrm.dataset import TestCollectionV2
from mmnrm.modelsv2 import shallow_interaction_model
from mmnrm.utils import set_random_seed, load_neural_model, save_model_weights, load_model, load_sentence_generator

import numpy as np

from timeit import default_timer as timer


import tensorflow as tf

# fix the generator randoness
set_random_seed()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('batch_size', type=int, help="size of the batch that will be used during the tests")
    parser.add_argument('-o', type=str, default=None, help="output file to append the results")

    args = parser.parse_args()
    
    # use bioASQ download folder 

    rank_model, _out = load_neural_model("download_folder/models/still-butterfly-1_batch0_map", return_snippets_score=False)
    test_input_generator = _out[1]

    def convert_to_tensor(data_generator):
        for query_id, Y, docs_info, offsets_docs in test_input_generator(data_generator):
            yield query_id, [tf.convert_to_tensor(Y[0], dtype=tf.int32), tf.convert_to_tensor(Y[1], dtype=tf.int32), tf.convert_to_tensor(Y[2], dtype=tf.int32)], docs_info, offsets_docs

    test_collection = TestCollectionV2\
                                    .load("download_folder/pickle-data/query_docs_pairs")\
                                    .batch_size(args.batch_size)\
                                    .set_transform_inputs_fn(convert_to_tensor)

    q_scores = defaultdict(list)

    times = []

    @tf.function
    def untraced_static_model(x):
        return rank_model(x)

    query_id, Y, docs_info, offsets_docs = next(test_collection.generator())
    static_model = untraced_static_model.get_concrete_function(Y)    
    static_model(*Y)

    for i, _out in enumerate(test_collection.generator()):
        query_id, Y, docs_info, offsets_docs = _out
        s_time = timer()

        #scores, q_sentence_attention = rank_model.predict(Y)
        output = static_model(*Y)
        dummy = output[0].numpy()
        times.append(timer()-s_time)

        if not i%10:
            print("Evaluation {} | time {:.3f} | avg {:.3f} +/- {:.3f} | median {:.3f}".format(i, 
                                                                                           times[-1], 
                                                                                           np.mean(times),
                                                                                           np.std(times),
                                                                                           np.median(times)),end="\r")

        #scores = scores[:,0].tolist()
        #q_scores[query_id].extend(list(zip(docs_ids,scores)))
        #for i in range(len(docs_info)):
        #    q_scores[query_id[i]].append((docs_info[i], scores[i], q_sentence_attention[i], offsets_docs[i]))

    # sort the rankings
    #for query_id in q_scores.keys():
    #    q_scores[query_id].sort(key=lambda x:-x[1])
    #    q_scores[query_id] = q_scores[query_id][:10]


    #times = times[2:len(times)-3]

    if args.o is not None:
        with open(args.o, "a") as f:
            f.write("{},{},{:.3f},{:.3f},{:.3f}\n".format("ours",
                                                     args.batch_size,
                                                     np.mean(times),
                                                     np.std(times),
                                                     np.median(times)))
