import argparse
import time

import os

from collections import defaultdict

from mmnrm.dataset import TrainCollectionV2, sentence_splitter_builderV2
from mmnrm.utils import set_random_seed, load_model_config, load_neural_model

from mmnrm.modelsv2 import shallow_interaction_model

import numpy as np

import tensorflow as tf

from mmnrm.callbacks import Callback
from mmnrm.training import PairwiseTraining

# In[2]:
class RecordTimes(Callback):
    def __init__(self, **kwargs):
        self.times=[]

    def on_step_end(self, training_obj, epoch, step, loss, time):
        self.times.append(time)
        if len(self.times)>2:
            print("\rEvaluation {} | time {:.3f} | avg {:.3f} ± {:.3f} | median {:.3f}".format(step,
                                                                                               self.times[-1],
                                                                                               np.mean(self.times[1:]),
                                                                                               np.std(self.times[1:]),
                                                                                               np.median(self.times[1:]),
                                                                                               end="\r"))


# fix the generator randoness
set_random_seed()

if __name__ == "__main__":
    
# use bioASQ download folder 

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('batch_size', type=int, help="size of the batch that will be used during the tests")
    parser.add_argument('-o', type=str, default=None, help="output file to append the results")

    args = parser.parse_args()

    rank_model, out = load_neural_model("download_folder/models/still-butterfly-1_batch0_map", return_snippets_score=False)
    train_generator = out[0]

    train_collection = TrainCollectionV2\
                        .load("download_folder/pickle-data/training_data_for_EACL")\
                        .batch_size(args.batch_size)\
                        .set_transform_inputs_fn(train_generator)






#model = tf.function(model)
#static_model = tf.function(model.call).get_concrete_function([tf.TensorSpec([None, INPUT_MAX_LENGTH], tf.int32, name="input_ids")])#, 
                                                              #tf.TensorSpec([None, INPUT_MAX_LENGTH], tf.int32, name="attention_mask")])


    callback_times = RecordTimes()
    

    @tf.function
    def clip_grads(grads):
        gradients, _ = tf.clip_by_global_norm(grads, 5.0)
        return gradients


    train = PairwiseTraining(model=rank_model,
                                 train_collection=train_collection,
                                 grads_callback=clip_grads,
                                 optimizer=tf.keras.optimizers.Adam(),
                                 callbacks=[callback_times])

                              
    train.train(1, draw_graph=False)   

    times = callback_times.times[1:]

    print("{} | avg {:.3f} +/- {:.3f} | median {:.3f}\n".format(BATCH_SIZE,
                                                         np.mean(times[:-1]),
                                                         np.std(times[:-1]),
                                                         np.median(times[:-1])))
    
    if args.o is not None:
        with open(args.o, "a") as f:
            f.write("{},{},{:.3f},{:.3f},{:.3f}\n".format("ours",
                                                     args.batch_size,
                                                     np.mean(times),
                                                     np.std(times),
                                                     np.median(times)))
