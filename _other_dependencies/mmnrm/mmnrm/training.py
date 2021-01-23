"""
This file contains an abstraction for implment pairwise loss training
"""

import tensorflow as tf
from tensorflow.keras import backend as K
import time
import tempfile
import shutil
import subprocess
import os
from collections import defaultdict

from timeit import default_timer as timer

from mmnrm.utils import save_model_weights, load_model_weights, set_random_seed, merge_dicts, flat_list, index_from_list
from mmnrm.text import TREC04_merge_goldstandard_files
from mmnrm.callbacks import WandBValidationLogger 

import random
import numpy as np
import pickle

import wandb
import nltk

def hinge_loss(positive_score, negative_score, *args):
    return K.mean(K.maximum(0., 1. - positive_score + negative_score))

def pairwise_cross_entropy(positive_score, negative_score, *args):
    positive_exp = K.exp(positive_score)
    return K.mean(-K.log(positive_exp/(positive_exp+K.exp(negative_score))))
    

class BaseTraining():
    def __init__(self, 
                 model,
                 loss,
                 train_collection,
                 optimizer="adam", # keras optimizer
                 callbacks=[],
                 **kwargs): 
        super(BaseTraining, self).__init__(**kwargs)
        self.model = model
        self.loss = loss
        
        self.train_collection = train_collection
        
        self.optimizer = tf.keras.optimizers.get(optimizer)
        
        self.callbacks = callbacks

    def draw_graph(self, name, *data):

        logdir = 'logs/func/'+name 
        writer = tf.summary.create_file_writer(logdir)

        tf.summary.trace_on(graph=True, profiler=True)

        self.training_step(*data)

        with writer.as_default():
            tf.summary.trace_export(
              name="training_trace",
              step=0,
              profiler_outdir=logdir)
            
    def train(self, epoch, draw_graph=True):
        raise NotImplementedError("This is an abstract class, should not be initialized")

class PairwiseTraining(BaseTraining):
    
    def __init__(self, loss=hinge_loss,  grads_callback=None, transform_model_inputs_callback=None, **kwargs):
        super(PairwiseTraining, self).__init__(loss=loss, **kwargs)
        self.grads_callback = grads_callback 
        self.transform_model_inputs_callback = transform_model_inputs_callback
    
    def predict_score(self, inputs):
        output = self.model.predict(inputs)
        
        if isinstance(self.model.output,list):
            return output[0]
        else:
            return output
    
    @tf.function # check if this can reutilize the computational graph for the prediction phase
    def model_score(self, inputs):
        print("\r[DEBUG] CALL MODEL_SCORE FUNCTION")
        return self.model(inputs)
    
    @tf.function # build a static computational graph
    def training_step(self, pos_in, neg_in, custom_output=None):
        print("training step")
        
        if self.transform_model_inputs_callback is not None:
            pos_in, neg_in, pos_label, neg_label = self.transform_model_inputs_callback(pos_in, neg_in)
        
        # manual optimization
        with tf.GradientTape() as tape:
            pos_score = self.model_score(pos_in)
            neg_score = self.model_score(neg_in)
            
            if custom_output is not None:
                print("DEBUG custom output")
                pos_score = custom_output(pos_score)
                neg_score = custom_output(neg_score)

            loss = self.loss(pos_score, neg_score, pos_label, neg_label)

        # using auto-diff to get the gradients
        grads = tape.gradient(loss, self.model.trainable_weights)
        
        #normalize grads???????????????????????????
        if self.grads_callback is not None:
            grads = self.grads_callback(grads)
        #tf.print(grads)
        # applying the gradients using an optimizer
        #tf.print(self.model.trainable_weights[-1])
        
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss

    def evaluate_test_set(self, test_set):
        generator_Y = test_set.generator()
                
        q_scores = defaultdict(list)

        for i, _out in enumerate(generator_Y):
            query_id, Y, docs_ids = _out
            s_time = time.time()
            scores = self.model_score(Y).numpy()[:,0].tolist()
            print("\rEvaluation {} | time {}".format(i, time.time()-s_time), end="\r")
            q_scores[query_id].extend(list(zip(docs_ids,scores)))

        # sort the rankings
        for query_id in q_scores.keys():
            q_scores[query_id].sort(key=lambda x:-x[1])

        # evaluate
        return test_set.evaluate(q_scores)
    
    def train(self, epoch, draw_graph=False, custom_output=None):
        
        # create train generator
        steps = self.train_collection.get_steps()
        generator_X = self.train_collection.generator()
        
        positive_inputs, negative_inputs = next(generator_X)
        
        if draw_graph:
            self.draw_graph(positive_inputs, negative_inputs)
        
        for c in self.callbacks:
            c.on_train_start(self)
        
        for e in range(epoch):
            
            #execute callbacks
            for c in self.callbacks:
                c.on_epoch_start(self, e)
                
            for s in range(steps):
                
                #execute callbacks
                for c in self.callbacks:
                    c.on_step_start(self, e, s)
                    
                s_time = timer()
                
                # static TF computational graph for the traning step
                loss = self.training_step(positive_inputs, negative_inputs, custom_output).numpy()
                
                #execute callbacks
                f_time = timer()-s_time
                for c in self.callbacks:
                    c.on_step_end(self, e, s, loss, f_time)
                    
                # next batch
                positive_inputs, negative_inputs = next(generator_X)
            
            #execute callbacks
            for c in self.callbacks:
                c.on_epoch_end(self, e)
                
        #execute callbacks
        for c in self.callbacks:
            c.on_train_end(self)      

class CrossValidation(BaseTraining):
    def __init__(self,
                 loss=hinge_loss,
                 wandb_config=None,
                 callbacks=[],
                 **kwargs):
        super(CrossValidation, self).__init__(loss=loss, **kwargs)
        self.wandb_config = wandb_config
        self.callbacks = callbacks
                
    def train(self, epoch, draw_graph=False):
        
        print("Start the Cross validation for", self.train_collection.get_n_folds(), "folds")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # save model init state
            save_model_weights(os.path.join(temp_dir,"temp_weights.h5"), self.model)
            best_test_scores = []
            for i, collections in enumerate(self.train_collection.generator()):
                print("Prepare FOLD", i)
                
                train_collection, test_collection = collections
                
                # show baseline metrics over the previous ranking order
                pre_metrics = test_collection.evaluate_pre_rerank()
                print("Evaluation of the original ranking order")
                for n,m in pre_metrics:
                    print(n,m)
                
                
                # reset all the states
                set_random_seed()
                K.clear_session()

                # load model init state
                load_model_weights(os.path.join(temp_dir,"temp_weights.h5"), self.model)
                
                self.wandb_config["name"] = "Fold_0"+str(i)+"_"+self.wandb_config["name"]
                
                # create evaluation callback
                if self.wandb_config is not None:
                    wandb_val_logger = WandBValidationLogger(wandb_args=self.wandb_config,
                                                             steps_per_epoch=train_collection.get_steps(),
                                                             validation_collection=test_collection)
                else:
                    raise KeyError("Please use wandb for now!!!")
                    
                best_test_scores.append(wandb_val_logger.current_best)
                    
                callbacks = [wandb_val_logger]+ self.callbacks
                
                print("Train and test FOLD", i)
                
                pairwise_training = PairwiseTraining(model=self.model,
                                                         train_collection=train_collection,
                                                         loss=self.loss,
                                                         optimizer=self.optimizer,
                                                         callbacks=callbacks)
                
                pairwise_training.train(epoch, draw_graph=draw_graph)
            
            x_score = sum(best_test_scores)/len(best_test_scores)
            print("X validation best score:", x_score)
            wandb_val_logger.wandb.run.summary["best_xval_"+wandb_val_logger.comparison_metric] = x_score
            
        except Exception as e:
            raise e # maybe handle the exception in the future
        finally:
            # always remove the temp directory
            print("Remove {}".format(temp_dir))
            shutil.rmtree(temp_dir)
        
         
# Create a more abstract class that uses common elemetns like, b_size, transform_input etc...

class BaseCollection:
    def __init__(self, 
                 transform_inputs_fn=None, 
                 b_size=64, 
                 name=None,
                 **kwargs):
        self.transform_inputs_fn = transform_inputs_fn
        self.b_size = b_size
        self.name = name
    
    def update_query_list(self, query_list):
        # NEED REFACTOR, class TEST and TRAIN collection that extend BaseCollection has query_list parameter, that should be moved to this class
        self.query_list = query_list
        return self
    
    def set_transform_inputs_fn(self, transform_inputs_fn):
        # build style method
        self.transform_inputs_fn = transform_inputs_fn
        return self
    
    def batch_size(self, b_size=32):
        # build style method
        self.b_size = b_size
        return self
    
    def get_config(self):
        data_json = {
            "b_size": self.b_size
        } 
        
        return data_json
    
    def set_name(self, name):
        self.name = name
        return self
    
    def generator(self, **kwargs):
        # generator for the query, pos and negative docs
        gen_X = self._generate(**kwargs)
        
        if self.transform_inputs_fn is not None:
            gen_X = self.transform_inputs_fn(gen_X)
        
        # finally yield the input to the model
        for X in gen_X:
            yield X
    
    def save(self, path):
        with open(path+".p", "wb") as f:
            pickle.dump(self.get_config(), f)
            
    @classmethod
    def load(cls, path):
        with open(path+".p", "rb") as f:
            config = pickle.load(f)
        
        return cls(**config)
    
class CrossValidationCollection(BaseCollection):
    """
    Helper class to store the folds data and build the respective Train and Test Collections
    """
    def __init__(self, 
                 folds_query_list,
                 folds_goldstandard,
                 folds_goldstandard_trec_file, 
                 folds_query_docs, 
                 trec_script_eval_path,
                 **kwargs):
        super(CrossValidationCollection, self).__init__(**kwargs)
        self.folds_query_list = folds_query_list 
        self.folds_goldstandard = folds_goldstandard
        self.folds_goldstandard_trec_file = folds_goldstandard_trec_file
        self.folds_query_docs = folds_query_docs
        self.trec_script_eval_path = trec_script_eval_path
        
        # assert fold size
    
    def get_n_folds(self):
        return len(self.folds_query_list)
    
    def _generate(self, **kwargs):

        for i in range(len(self.folds_query_list)):
            # create the folds
            test_query = self.folds_query_list[i]
            test_goldstandard_trec_file = self.folds_goldstandard_trec_file[i]
            test_query_docs = self.folds_query_docs[i]

            train_query = flat_list(self.folds_query_list[:i] + self.folds_query_list[i+1:])
            train_goldstandard = merge_dicts(self.folds_goldstandard[:i] + self.folds_goldstandard[i+1:])
            train_query_docs = merge_dicts(self.folds_query_docs[:i] + self.folds_query_docs[i+1:])
            
            train_collection = TrainCollection(train_query, train_goldstandard, train_query_docs)
            
            test_collection = TestCollection(test_query, test_goldstandard_trec_file, test_query_docs, self.trec_script_eval_path, train_collection.skipped_queries)
            

            yield train_collection, test_collection

    def get_config(self):
        super_config = super().get_config()
        
        data_json = {
            "folds_query_list": self.folds_query_list,
            "folds_goldstandard": self.folds_goldstandard,
            "folds_goldstandard_trec_file": self.folds_goldstandard_trec_file,
            "folds_query_docs": self.folds_query_docs,
            "trec_script_eval_path": self.trec_script_eval_path
        } 
        
        return dict(data_json, **super_config) #fast dict merge
        
class TestCollection(BaseCollection):
    def __init__(self, 
                 query_list,
                 query_docs, 
                 evaluator,
                 skipped_queries = [],
                 **kwargs):
        """
        query_list - must be a list with the following format :
                     [
                         {
                             id: <str>
                             query: <str>
                         },
                         ...
                     ]
                       
        query_docs  - dictionary with documents to be ranked by the model
                       {
                           id: [{
                               id: <str>
                               text: <str>
                           }],
                           ...
                       }
                       
        """
        super(TestCollection, self).__init__(**kwargs)
        self.query_list = query_list 
        self.query_docs = query_docs
        self.evaluator = evaluator
        
        self.skipped_queries = skipped_queries
        
        if isinstance(self.evaluator, dict):
            self.evaluator = self.evaluator["class"].load(**self.evaluator)

    def get_config(self):
        super_config = super().get_config()
        
        data_json = {
            "query_list": self.query_list,
            "query_docs": self.query_docs,
            "skipped_queries": self.skipped_queries,
            "evaluator": self.evaluator.get_config()
        } 
        
        return dict(data_json, **super_config) #fast dict merge
    
    def _generate(self, **kwargs):
        
        for query_data in self.query_list:
            if query_data["id"] in self.skipped_queries:
                continue
                
            for i in range(0, len(self.query_docs[query_data["id"]]), self.b_size):
                docs = self.query_docs[query_data["id"]][i:i+self.b_size]
                
                yield query_data["id"], query_data["query"], docs
    
    def evaluate_pre_rerank(self, output_metris=["recall_100", "map_cut_20", "ndcg_cut_20", "P_20"]):
        """
        Compute evaluation metrics over the documents order before been ranked
        """ 
        ranked_format = {k:list(map(lambda x:(x[1]["id"], len(v)-x[0]), enumerate(v))) for k,v in self.query_docs.items()}
        
        metrics = self.evaluate(ranked_format)
        
        if isinstance(output_metris, list):
            return [ (m, metrics[m]) for m in output_metris]
        else:
            return metrics
    
    def evaluate_oracle(self, output_metris=["recall_100", "map_cut_20", "ndcg_cut_20", "P_20"]):
        metrics = self.evaluator.evaluate_oracle()
    
        if isinstance(output_metris, list):
            return [ (m, metrics[m]) for m in output_metris]
        else:
            return metrics

    def evaluate(self, ranked_query_docs):
        return self.evaluator.evaluate(ranked_query_docs)
    
    

class TrainCollection(BaseCollection):
    def __init__(self, 
                 query_list, 
                 goldstandard, 
                 query_docs_subset=None, 
                 use_relevance_groups=False,
                 verbose=True, 
                 **kwargs):
        """
        query_list - must be a list with the following format :
                     [
                         {
                             id: <str>
                             query: <str>
                         },
                         ...
                     ]
        
        goldstandard - must be a dictionary with the following format:
                       {
                           id: {
                               0: [<str:id>, <str:id>, ...],
                               1: [<str:id>, <str:id>, ...],
                               2: ...,
                               ...
                           },
                           ...
                       }
                       
        query_docs_subset (optional) - if a previous retrieved method were used to retrieved the TOP_K documents, this parameter
                                       can be used to ignore the collection and instead use only the TOP_K docs
                                       {
                                           id: [{
                                               id: <str>
                                               text: <str>
                                           }, ...],
                                           ...
                                       }
        """
        super(TrainCollection, self).__init__(**kwargs)
        self.query_list = query_list # [{query data}]
        self.goldstandard = goldstandard # {query_id:[relevance docs]}
        self.use_relevance_groups = use_relevance_groups
        self.verbose = verbose
        
        if "sub_set_goldstandard" in kwargs:
            self.sub_set_goldstandard = kwargs.pop("sub_set_goldstandard")
        else:
            self.sub_set_goldstandard = None
        
        if "collection" in kwargs:
            self.collection = kwargs.pop("collection")
        else:
            self.collection = None
        
        self.skipped_queries = []

        self.__build(query_docs_subset)
    
    def __find_relevance_group(self, doc_id, search_gs):
        for k in search_gs.keys():
            if doc_id in search_gs[k]:
                return k
        return -1
    
    def __build(self, query_docs_subset):
        
        if query_docs_subset is None:
            # number of samples
            return #
        
        self.sub_set_goldstandard = {}
        self.collection = {}
  
        # filter the goldstandard
        for _id, relevance in query_docs_subset.items():
            
            if _id not in self.goldstandard:
                self.skipped_queries.append(_id)
                continue
            
            # do not use queries without true positives
            # this add an overhead that can be avoided by refactor the follwing for loop!
            unique_relevants = set(sum([self.goldstandard[_id][k] for k in self.goldstandard[_id].keys() if k>0], []))
            if all([ doc["id"] not in unique_relevants for doc in relevance ]):
                self.skipped_queries.append(_id)
                continue
            
            self.sub_set_goldstandard[_id] = defaultdict(list)
            
            for doc in relevance:
                k = self.__find_relevance_group(doc["id"], self.goldstandard[_id])
                if k>0:
                    if self.use_relevance_groups:
                        self.sub_set_goldstandard[_id][k].append(doc["id"])
                    else:
                        self.sub_set_goldstandard[_id][1].append(doc["id"])
                else:
                    # default add to the less relevance group
                    self.sub_set_goldstandard[_id][0].append(doc["id"])
                
                #add to the collection
                self.collection[doc["id"]] = doc["text"]
        
        # remove the skipped queries from the data
        index_to_remove = []
        
        for skipped in self.skipped_queries:
            _index = index_from_list(self.query_list, lambda x: x["id"]==skipped)
            if _index>-1:
                index_to_remove.append(_index)
        index_to_remove.sort(key=lambda x:-x)
        
        # start removing from the tail
        for _index in index_to_remove:
            del self.query_list[_index]
        
        # stats
        if self.verbose:
            max_keys = max(map(lambda x:max(x.keys()), self.sub_set_goldstandard.values()))
            
            for k in range(max_keys+1):
                print("Minimum number of relevance type({}) in the queries of the goldstandard sub set: {}".format(k, min(map(lambda x: len(x[k]), self.sub_set_goldstandard.values()))))
            
                print("Mean number of relevance type({}) in the queries of the goldstandard sub set: {}".format(k, sum(map(lambda x: len(x[k]), self.sub_set_goldstandard.values()))/len(self.sub_set_goldstandard)))
            
            print("Sub Collection size", len(self.collection))
            print("Number of skipped question, due to lack of true positives", len(self.skipped_queries))
    
    def __get_goldstandard(self):
        
        if self.collection is not None:
            return self.sub_set_goldstandard
        else:
            return self.goldstandard
    
    def get_steps(self):
        
        training_data = self.__get_goldstandard()
        
        # an epoch will be defined with respect to the total number of positive pairs
        total_positives = sum(map(lambda x: sum([ len(x[k]) for k in x.keys() if k>0]), training_data.values()))
          
        return total_positives//self.b_size

    def _generate(self, collection=None, **kwargs):
        
        # sanity check
        assert(not(self.sub_set_goldstandard==None and collection==None))
        
        training_data = self.__get_goldstandard()
        
        # TODO this condition is dependent on the previous
        if collection is None:
            collection = self.collection
            
        while True:
            # TODO check if it is worthit to use numpy to vectorize these operations
            
            y_query = []
            y_pos_doc = []
            y_neg_doc = []
            
            # build $batch_size triples and yield
            query_indexes = random.sample(population=list(range(len(self.query_list))), k=self.b_size)
            for q_i in query_indexes:
                selected_query = self.query_list[q_i]
                #print(selected_query["id"])
                # select the relevance group, (only pos)
                positive_keys=list(training_data[selected_query["id"]].keys())
                #print("positive_keys", positive_keys)
                positive_keys.remove(0)
                #print("positive_keys", positive_keys)
                if len(positive_keys)>1:
                    group_len = list(map(lambda x: len(training_data[selected_query["id"]][x]), positive_keys))
                    total = sum(group_len)
                    prob = list(map(lambda x: x/total, group_len))
                    #print("probs", prob)
                    relevance_group = np.random.choice(positive_keys, p=prob)
                else:
                    relevance_group = positive_keys[0]
                
                _pos_len = len(training_data[selected_query["id"]][relevance_group])
                pos_doc_index = random.randint(0, _pos_len-1) if _pos_len>1 else 0
                pos_doc_id = training_data[selected_query["id"]][relevance_group][pos_doc_index]
                pos_doc = collection[pos_doc_id]
                
                _neg_len = len(training_data[selected_query["id"]][relevance_group-1])
                neg_doc_index = random.randint(0, _neg_len-1) if _neg_len>1 else 0
                neg_doc_id = training_data[selected_query["id"]][relevance_group-1][neg_doc_index]
                neg_doc = collection[neg_doc_id]
                
                y_query.append(selected_query["query"])
                y_pos_doc.append(pos_doc)
                y_neg_doc.append(neg_doc)
            
            yield (np.array(y_query), np.array(y_pos_doc), np.array(y_neg_doc))
    
    def get_config(self):
        super_config = super().get_config()
        
        data_json = {
            "query_list": self.query_list,
            "goldstandard": self.goldstandard,
            "use_relevance_groups": self.use_relevance_groups,
            "verbose": self.verbose,
            "sub_set_goldstandard": self.sub_set_goldstandard,
            "collection": self.collection,
        } 
        
        return dict(data_json, **super_config) #fast dict merge
    

    
def sentence_splitter_builder(tokenizer, mode=0, max_sentence_size=21):
    """
    Return a transform_inputs_fn for training and test as a tuple
    
    mode 0: use fixed sized window for the split
    mode 1: split around a query-document match with a fixed size
    mode 2: deeprank alike. Similar to mode 1, but group the match by q-term
    mode 3: split with ntlk sentence splitter
    mode 4: similar to 2, but uses sentence splitting instead of fix size
    """
    if mode in [1, 2]:
        half_window = max_sentence_size//2
        min_w = lambda x: max(0,x-half_window)
        max_w = lambda x,l: min(x+half_window,l)+1
    
    def train_splitter(data_generator):

        while True:
        
            # get the batch triplet
            query, pos_docs, neg_docs = next(data_generator)
            
            # tokenization
            query = tokenizer.texts_to_sequences(query)
            
            if mode not in [3, 4]: # for the mode 3 this is a preprocessing step
                pos_docs = tokenizer.texts_to_sequences(pos_docs)
                neg_docs = tokenizer.texts_to_sequences(neg_docs)

                if any([ len(doc)==0 for doc in pos_docs]):
                    continue # try a new resampling, NOTE THIS IS A EASY FIX PLS REDO THIS!!!!!!!
                             # for obvious reasons

            new_pos_docs = []
            new_neg_docs = []
            
            # sentence splitting
            if mode==0:
                
                for i in range(len(pos_docs)):
                    new_pos_docs.append([])
                    new_neg_docs.append([])
                    for s in range(0, len(pos_docs[i]), max_sentence_size):
                        new_pos_docs[-1].append(pos_docs[i][s:s+max_sentence_size])
                    for s in range(0, len(neg_docs[i]), max_sentence_size):
                        new_neg_docs[-1].append(neg_docs[i][s:s+max_sentence_size])
            elif mode==1:

                for b in range(len(pos_docs)):
                    new_pos_docs.append([])
                    new_neg_docs.append([])
                    # split by exact matching
                    for t_q in query[b]:
                        # exact math for the pos_document
                        for i,t_pd in enumerate(pos_docs[b]):
                            if t_pd==t_q:
                                new_pos_docs[-1].append(pos_docs[b][min_w(i):max_w(i,len(pos_docs[b]))])

                        # exact math for the neg_document
                        for i,t_nd in enumerate(neg_docs[b]):
                            if t_nd==t_q:
                                new_neg_docs[-1].append(neg_docs[b][min_w(i):max_w(i,len(neg_docs[b]))])
            elif mode==2:
                
                for b in range(len(pos_docs)):
                    new_pos_docs.append([])
                    new_neg_docs.append([])
                    # split by exact matching
                    for t_q in query[b]:
                        # entry for the query-term
                        new_pos_docs[-1].append([])
                        new_neg_docs[-1].append([])
                        
                        # exact math for the pos_document
                        for i,t_pd in enumerate(pos_docs[b]):
                            if t_pd==t_q:
                                new_pos_docs[-1][-1].append(pos_docs[b][min_w(i):max_w(i,len(pos_docs[b]))])

                        # exact math for the neg_document
                        for i,t_nd in enumerate(neg_docs[b]):
                            if t_nd==t_q:
                                new_neg_docs[-1][-1].append(neg_docs[b][min_w(i):max_w(i,len(neg_docs[b]))])
            elif mode==3:
                
                for b in range(len(pos_docs)):
                    new_pos_docs.append([])
                    new_neg_docs.append([])
                    
                    for pos_sentence in nltk.sent_tokenize(pos_docs[b]):
                        new_pos_docs[-1].append(pos_sentence)
                    for neg_sentence in nltk.sent_tokenize(neg_docs[b]):
                        new_neg_docs[-1].append(neg_sentence)
                    
                    new_pos_docs[-1] = tokenizer.texts_to_sequences(new_pos_docs[-1])
                    new_neg_docs[-1] = tokenizer.texts_to_sequences(new_neg_docs[-1])
            elif mode==4:
                
                for b in range(len(pos_docs)):
                    new_pos_docs.append([])
                    new_neg_docs.append([])
                    
                    _temp_pos_docs = nltk.sent_tokenize(pos_docs[b])
                    _temp_pos_docs = tokenizer.texts_to_sequences(_temp_pos_docs)
                    
                    _temp_neg_docs = nltk.sent_tokenize(neg_docs[b])
                    _temp_neg_docs = tokenizer.texts_to_sequences(_temp_neg_docs)
                    
                    # split by exact matching
                    for t_q in query[b]:
                        # entry for the query-term
                        new_pos_docs[-1].append([])
                        new_neg_docs[-1].append([])
                        
                        for pos_sent in _temp_pos_docs:
                            # exact math for the pos_document
                            for i,t_pd in enumerate(pos_sent):
                                if t_pd==t_q:
                                    new_pos_docs[-1][-1].append(pos_sent)
                                    break

                        for neg_sent in _temp_neg_docs:
                            for i,t_nd in enumerate(neg_sent):
                                if t_nd==t_q:
                                    new_neg_docs[-1][-1].append(neg_sent)
                                    break
            else:
                raise NotImplementedError("Missing implmentation for mode "+str(mode))

            yield query, new_pos_docs, new_neg_docs
            
            
    def test_splitter(data_generator):

        for _id, query, docs in data_generator:

            # tokenization
            tokenized_query = tokenizer.texts_to_sequences([query])[0]
            for doc in docs:
                if isinstance(doc["text"], list):
                    continue # cached tokenization
                    
                if mode not in [3, 4]:
                    doc["text"] = tokenizer.texts_to_sequences([doc["text"]])[0]

                # sentence splitting
                new_docs = []
                if mode==0:
                    for s in range(0,len(doc["text"]), max_sentence_size):
                        new_docs.append(doc["text"][s:s+max_sentence_size])
                elif mode==1:
                    for t_q in tokenized_query:
                        for i,t_d in enumerate(doc["text"]):
                            if t_d==t_q:
                                new_docs.append(doc["text"][min_w(i):max_w(i,len(doc["text"]))])
                elif mode==2:
                    for t_q in tokenized_query:
                        new_docs.append([])
                        for i,t_d in enumerate(doc["text"]):
                            if t_d==t_q:
                                new_docs[-1].append(doc["text"][min_w(i):max_w(i,len(doc["text"]))])
                elif mode==3:
                    for s in nltk.sent_tokenize(doc["text"]):
                        new_docs.append(s)
                    new_docs = tokenizer.texts_to_sequences(new_docs)
                elif mode==4:
                    _temp_new_docs = tokenizer.texts_to_sequences(nltk.sent_tokenize(doc["text"]))
                    for t_q in tokenized_query:
                        new_docs.append([])
                        for _new_doc in _temp_new_docs:
                            for i,t_d in enumerate(_new_doc):
                                if t_d==t_q:
                                    new_docs[-1].append(_new_doc)
                                    break
                else:
                    raise NotImplementedError("Missing implmentation for mode "+str(mode))
                                                                    
                doc["text"] = new_docs
                                                                    
            yield _id, tokenized_query, docs

    return train_splitter, test_splitter


