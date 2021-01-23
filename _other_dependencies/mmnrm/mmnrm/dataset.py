import time
import tempfile
import shutil
import subprocess
import os
from collections import defaultdict

from mmnrm.utils import save_model_weights, load_model_weights, set_random_seed, merge_dicts, flat_list, index_from_list, overlap

import random
import numpy as np
import pickle
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
import math

from mmnrm.training import BaseCollection

class TrainPairwiseCollection(BaseCollection):
    """
    Follows a TREC like style, were a set of integer relevance is given (0,1,2,...)
    """
    
    def __init__(self, 
                 query_list, 
                 goldstandard,
                 collection,
                 query_sampling_strategy = 0,
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
        collection - dictionary
                    {
                        <str:id>:<str:doc text>
                    }
                    
        """
        super(TrainPairwiseCollection, self).__init__(**kwargs)
        self.query_list = query_list # [{query data}]
        self.set_goldstandard(goldstandard) # {query_id:[relevance docs]}
        self.collection = collection
        self.set_query_sampling_strategy(query_sampling_strategy)

    
    def set_goldstandard(self, goldstandard):
        # cache the original gs
        self.goldstandard = goldstandard
        
        self.goldstandard_keys_by_relevance = {}
        
        for _id in goldstandard.keys():
            
            # do some precomputation
            self.goldstandard_keys_by_relevance[_id] = list(self.goldstandard[_id].keys())
            self.goldstandard_keys_by_relevance[_id].sort()

            # only use the positive keys
            # self.goldstandard_keys_by_relevance[_id] = self.goldstandard_keys_by_relevance[_id][1:]
    
    def set_query_sampling_strategy(self, strategy):
        """
        strategy: 0 - random by id
                  1 - uniform distribution that acouts the number of positive feedback
                  2 - uniform distribution that acouts for the total number of feedback (includes negative)
        """
        self.query_sampling_strategy = strategy
        
        if strategy==1:
            # do some precomputation
            self.gs_positive_total_docs = { _id:sum( [ len(self.goldstandard[_id][relevance_id]) for relevance_id in self.goldstandard_keys_by_relevance[_id][1:]]) for _id in self.goldstandard.keys()}
            self.query_ids = [ q_data["id"] for q_data in self.query_list]
        elif strategy==2:
            # do some precomputation
            self.gs_total_docs = { _id:sum( [ len(self.goldstandard[_id][relevance_id]) for relevance_id in self.goldstandard_keys_by_relevance[_id]]) for _id in self.goldstandard.keys()}
            self.query_ids = [ q_data["id"] for q_data in self.query_list]
        
        return self
    
    def __linear_probs(self, groups, selected_gs, use_len=True, b_size=None, return_indexes=False):
        
        if use_len:
            group_len = list(map(lambda x: len(selected_gs[x]), groups))
        else:
            group_len = list(map(lambda x: selected_gs[x], groups))
            
        total = sum(group_len)
        prob = list(map(lambda x: x/total, group_len))
        
        if return_indexes:
            groups = range(len(prob)) # same as range(len(groups))
            
        return np.random.choice(groups, p=prob, size=b_size)
    
    
    def __choose_query(self):

        if self.query_sampling_strategy==0:
            return np.random.randint(0,len(self.query_list), size=self.b_size)
        elif self.query_sampling_strategy==1:
            return self.__linear_probs(self.query_ids, 
                                       self.gs_positive_total_docs, 
                                       use_len=False, 
                                       b_size=self.b_size,
                                       return_indexes=True)
        elif self.query_sampling_strategy==2:
            return self.__linear_probs(self.query_ids, 
                                       self.gs_total_docs, 
                                       use_len=False, 
                                       b_size=self.b_size,
                                       return_indexes=True)
        else:
            raise ValueError("query_sampling_strategy value {} is invalid".format(self.query_sampling_strategy))    
    
    def __choose_pos_neg_index(self, goldstandard_keys_by_relevance, selected_gs):
        
        goldstandard_positives = goldstandard_keys_by_relevance[1:]
        relevance_group = self.__linear_probs(goldstandard_positives, selected_gs)
        
        pos_index = random.randint(0, len(selected_gs[relevance_group])-1)
        
        # select the random also based on probabilities TODO
        
        goldstandard_negatives = [ neg_group for neg_group in goldstandard_keys_by_relevance if neg_group<relevance_group]
        less_relevance_group = self.__linear_probs(goldstandard_negatives, selected_gs) 
        
        neg_index = random.randint(0, len(selected_gs[less_relevance_group])-1)

        assert(relevance_group>less_relevance_group)
        
        pos_doc_id = selected_gs[relevance_group][pos_index]
        neg_doc_id = selected_gs[less_relevance_group][neg_index]
        return pos_doc_id, neg_doc_id

    
    def _generate(self):
        
            
        while True:
            # TODO check if it is worthit to use numpy to vectorize these operations
            
            y_query = []
            y_pos_doc = []
            y_neg_doc = []
            
            # build $batch_size triples and yield
            query_indexes = self.__choose_query()

            for q_i in query_indexes:
                selected_query = self.query_list[q_i]
                
                pos_doc_id, neg_doc_id = self.__choose_pos_neg_index(self.goldstandard_keys_by_relevance[selected_query["id"]],
                                                                     self.goldstandard[selected_query["id"]])
                
                pos_doc = {"text":self.collection[pos_doc_id]["text"]}
                neg_doc = {"text":self.collection[neg_doc_id]["text"]}
                
                y_query.append(selected_query["query"])
                y_pos_doc.append(pos_doc)
                y_neg_doc.append(neg_doc)
            
            yield (np.array(y_query), np.array(y_pos_doc), np.array(y_neg_doc))
            
    def get_steps(self):
        
        total_feedback = None # at the moment this var will be override
        
        if self.query_sampling_strategy==2:
            total_feedback = sum(map(lambda x: sum([ len(x[k]) for k in x.keys()]), self.goldstandard.values()))
        else:
            # an epoch will be defined with respect to the total number of positive pairs
            total_feedback = sum(map(lambda x: sum([ len(x[k]) for k in x.keys() if k>0]), self.goldstandard.values()))
          
        return total_feedback//self.b_size
    
    def get_config(self):
        super_config = super().get_config()
        
        data_json = {
            "query_list": self.query_list,
            "goldstandard": self.goldstandard,
            "collection": self.collection,
            "query_sampling_strategy": self.query_sampling_strategy,
        } 
        
        return dict(data_json, **super_config) #fast dict merge
    
class TrainCollectionV2(BaseCollection):
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
                                               score: <float>
                                           }, ...],
                                           ...
                                       }
        """
        super(TrainCollectionV2, self).__init__(**kwargs)
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
                        self.sub_set_goldstandard[_id][k].append({"id":doc["id"],"score":doc["score"]})
                    else:
                        self.sub_set_goldstandard[_id][1].append({"id":doc["id"],"score":doc["score"]})
                else:
                    # default add to the less relevance group
                    self.sub_set_goldstandard[_id][0].append({"id":doc["id"],"score":doc["score"]})
                
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
                pos_doc = {"text":collection[pos_doc_id["id"]], "score":pos_doc_id["score"]}
                
                _neg_len = len(training_data[selected_query["id"]][relevance_group-1])
                neg_doc_index = random.randint(0, _neg_len-1) if _neg_len>1 else 0
                neg_doc_id = training_data[selected_query["id"]][relevance_group-1][neg_doc_index]
                neg_doc = {"text":collection[neg_doc_id["id"]], "score":neg_doc_id["score"]}
                
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

    
class TestCollectionV2(BaseCollection):
    def __init__(self, 
                 query_list,
                 query_docs, 
                 evaluator=None,
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
                               score: <float>
                           }],
                           ...
                       }
                       
        """
        super(TestCollectionV2, self).__init__(**kwargs)
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
        
        query_ids = []
        queries = []
        query_docs = []
        i=0
        
        for query_data in self.query_list:

            if query_data["id"] in self.skipped_queries:
                continue
            if query_data["id"] not in self.query_docs:
                print("[WARNING] -",query_data["id"],"does not have docs, so it will be skipped")
                continue
            
            while True: #do while
                
                left_space = self.b_size-len(flat_list(query_docs))
                if len(self.query_docs[query_data["id"]][i:])<left_space:
                    # all the documents fit the batch
                    query_docs.append(self.query_docs[query_data["id"]][i:])
                    i=0
                else:
                    # docs do not fit in the batch
                    query_docs.append(self.query_docs[query_data["id"]][i:i+left_space])
                    i = i+left_space
                
                query_ids.append(query_data["id"])
                queries.append(query_data["query"])
                
                # DEBUG PRINTTTTT
                #print(query_data["id"], i, len(flat_list(query_docs)))
                
                
                #ouptup accoring to the batch size
                if len(flat_list(query_docs))>=self.b_size:
                    yield query_ids, queries, query_docs
                    # reset vars
                    query_ids = []
                    queries = []
                    query_docs = []
                
                
                if i==0:
                    break

                

            
    
    def evaluate_pre_rerank(self, output_metris=["recall_100", "map_cut_20", "ndcg_cut_20", "P_20"]):
        """
        Compute evaluation metrics over the documents order before been ranked
        """ 
        ranked_format = {k:list(map(lambda x:(x[1]["id"], len(v)-x[0]), enumerate(v))) for k,v in self.query_docs.items()}
        
        metrics = self.evaluate(ranked_format)
        
        if isinstance(output_metris, list):
            return { m:metrics[m] for m in output_metris}
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

def compute_extra_features(query_tokens, tokenized_sentences_doc, idf_fn):
    
    doc_tokens = sum(tokenized_sentences_doc, [])
    
    bi_gram_doc_tokens = set([(doc_tokens[i],doc_tokens[i+1]) for i in range(len(doc_tokens)-1)])
    bi_gram_query_tokens = set([(query_tokens[i],query_tokens[i+1]) for i in range(len(query_tokens)-1)])
    
    doc_tokens = set(doc_tokens)
    query_tokens = set(query_tokens)
    
    # compute percentage of q-terms in D
    num_dt_in_Q = len([ x for x in doc_tokens if x in query_tokens])
    num_Q = len(query_tokens)
    
    qt_in_D = num_dt_in_Q/num_Q
    
    # compute the weighted percentage of q-terms in D
    w_dt_in_Q = sum([ idf_fn(x) for x in doc_tokens if x in query_tokens])
    w_Q = sum([idf_fn(x) for x in query_tokens])
    
    W_qt_in_D = w_dt_in_Q/w_Q
    
    # compute the percentage of bigrams matchs in D
    num_bi_dt_in_bi_Q = len([ x for x in bi_gram_doc_tokens if x in bi_gram_query_tokens])
    num_bi_Q = len(bi_gram_query_tokens)
    
    bi_qt_in_bi_D = num_bi_dt_in_bi_Q/num_bi_Q
    
    return [qt_in_D, W_qt_in_D, bi_qt_in_bi_D]
    
    
    
def sentence_splitter_builderV2(tokenizer, mode=4, max_sentence_size=21, queries_sw=None, docs_sw=None, save_tokenizer_on_test=True):
    """
    Return a transform_inputs_fn for training and test as a tuple
    
    For now only the mode 4 is supported since it was the best from the previous version!
    
    mode 4: similar to 2, but uses sentence splitting instead of fix size
    
    queries_sw: set with sw for queries
    docs_sw: set with sw for documents
    """
    idf_from_id_token = lambda x: math.log(tokenizer.document_count/tokenizer.word_docs[tokenizer.index_word[x]])
    
    def train_splitter(data_generator):

        while True:
        
            # get the batch triplet
            query, pos_docs, neg_docs = next(data_generator)
            
            # tokenization
            query = tokenizer.texts_to_sequences(query)
            
            if queries_sw is not None:
                for tokenized_query in query:
                    tokenized_query = [token for token in tokenized_query if token not in queries_sw] 
            
            new_pos_docs = []
            new_neg_docs = []
            
            new_pos_extra_features = []
            new_neg_extra_features = []
            
            # sentence splitting
            if mode==4:
                
                for b in range(len(pos_docs)):
                    new_pos_docs.append([])
                    new_neg_docs.append([])
                    
                    _temp_pos_docs = nltk.sent_tokenize(pos_docs[b]["text"])
                    _temp_pos_docs = tokenizer.texts_to_sequences(_temp_pos_docs)
                    
                    if docs_sw is not None:
                        for tokenized_docs in _temp_pos_docs:
                            tokenized_docs = [token for token in tokenized_docs if token not in docs_sw] 
                    
                    # skip batch with empty pos_docs
                    if all([ len(sentence)==0  for sentence in _temp_pos_docs]):
                        break # try a new resampling, NOTE THIS IS A EASY FIX PLS REDO THIS!!!!!!!
                                 # for obvious reasons
                    
                    _temp_neg_docs = nltk.sent_tokenize(neg_docs[b]["text"])
                    _temp_neg_docs = tokenizer.texts_to_sequences(_temp_neg_docs)
                    
                    if docs_sw is not None:
                        for tokenized_docs in _temp_neg_docs:
                            tokenized_docs = [token for token in tokenized_docs if token not in docs_sw] 
  
                    # compute extra features
                    #extra_features_pos_doc = compute_extra_features(query[b], _temp_pos_docs, idf_from_id_token)
                    #extra_features_neg_doc = compute_extra_features(query[b], _temp_neg_docs, idf_from_id_token)
                    
                    # add the bm25 score
                    #extra_features_pos_doc.append(pos_docs[b]["score"])
                    #extra_features_neg_doc.append(neg_docs[b]["score"])
                    
                    # add all the extra features
                    #new_pos_extra_features.append(extra_features_pos_doc)
                    #new_neg_extra_features.append(extra_features_neg_doc)
                    
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
            
            if len(new_pos_docs) == len(pos_docs): # if batch is correct
                yield query, new_pos_docs, new_pos_extra_features, new_neg_docs, new_neg_extra_features
            
            
    def test_splitter(data_generator):

        for _id, query, docs in data_generator:
            tokenized_queries = []
            for i in range(len(_id)):
                # tokenization
                tokenized_query = tokenizer.texts_to_sequences([query[i]])[0]

                if queries_sw is not None:
                    tokenized_query = [token for token in tokenized_query if token not in queries_sw] 
                
                if save_tokenizer_on_test:
                    tokenized_queries.append(tokenized_query)
                else:
                    tokenized_queries.append(query[i]) # the tokenization isnt save
                    
        
                for doc in docs[i]:
                    if isinstance(doc["text"], list):
                        continue # cached tokenization

                    # sentence splitting
                    new_docs = []
                    if mode==4:
                        _temp_new_docs = []
                        doc["offset"] = []
                        doc["untokenized_text"] = doc["text"]
                        for start, end in PunktSentenceTokenizer().span_tokenize(doc["text"]):
                            _temp_new_docs.append(doc["text"][start:end])

                            if start<(len(doc["title"])-1):
                                doc["offset"].append(["title",(start, end), doc["text"][start:end], []])
                            else:
                                doc["offset"].append(["abstract", (start-len(doc["title"]), end-len(doc["title"])), doc["text"][start:end], []])
                        
                        tokenized_docs = tokenizer.texts_to_sequences(_temp_new_docs)

                        if docs_sw is not None:
                            for t_doc in tokenized_docs:
                                t_doc = [token for token in t_doc if token not in docs_sw]

                        #doc["extra_features"] = compute_extra_features(tokenized_query, _temp_new_docs, idf_from_id_token)+[doc["score"]]

                        for k,t_q in enumerate(tokenized_query):
                            new_docs.append([])
                            for l,_new_doc in enumerate(tokenized_docs):
                                for i,t_d in enumerate(_new_doc):
                                    if t_d==t_q:
                                        if save_tokenizer_on_test:
                                            new_docs[-1].append(_new_doc)
                                        else:
                                            new_docs[-1].append(_temp_new_docs[l])
                                        doc["offset"][l][-1].append(k)
                                        break
                    else:
                        raise NotImplementedError("Missing implmentation for mode "+str(mode))

                    doc["text"] = new_docs
                
                                                                    
            yield _id, tokenized_queries, docs

    return train_splitter, test_splitter



class TrainSnippetsCollection(BaseCollection):
    def __init__(self, 
                 query_list, 
                 goldstandard, 
                 query_docs_subset = None,
                 find_relevant_snippets = None, # default use the self contained function 
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
                           id: [{
                                    id: <str>,
                                    snippets: [{<bioasq snippet data>}
                                 }, ...],
                           ...
                       }
                       
        query_docs_subset (optional) - previous retrieved method of the retrieved the TOP_K documents
                                       {
                                           id: [{
                                               id: <str>
                                               snippets: [<str>, <str>]
                                               score: <float>
                                           }, ...],
                                           ...
                                       }
        """
        super(TrainSnippetsCollection, self).__init__(**kwargs)
        self.query_list = query_list # [{query data}]
        self.goldstandard = goldstandard # {query_id:[relevance docs]}
        
        # default self contained function association
        self.find_relevant_snippets = self.__find_relevant_snippets if find_relevant_snippets is None else find_relevant_snippets
        
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
    
    def __find_relevant_snippets(self, doc_to_sentence, gs_doc_snippets, title_len, doc_id):
        positive_snippet_index = [ 0 for _ in range(len(doc_to_sentence))]
        
        # a snippet is relevant if it contais some portion of the gs_snippet
        for gs_doc_id in gs_doc_snippets:
            if doc_id==gs_doc_id["id"]:
                for snippet in gs_doc_id["snippets"]:
                    # find the matching snippet
                    for index, doc_snippet in enumerate(doc_to_sentence):
                        if snippet["beginSection"] == "title":
                            _overlap = overlap((snippet["offsetInBeginSection"], snippet["offsetInEndSection"]), 
                                               (doc_snippet["start"], doc_snippet["end"]))
                        else:
                            _overlap = overlap((snippet["offsetInBeginSection"], snippet["offsetInEndSection"]), 
                                               (doc_snippet["start"], doc_snippet["end"]))

                        if _overlap > 0: # this snippet contains relevant information
                            positive_snippet_index[index] = 1
                            # break no 
                            # add break here?
                        
        # return in onehot encodding
        return positive_snippet_index
    
    def __build(self, query_docs_subset):
        
        if query_docs_subset is None:
            return 
        
        self.sub_set_goldstandard = {}
        self.collection = {}
        progress = 0
        # filter the goldstandard
        for _id, relevance in query_docs_subset.items():
            print("running query:", progress, end="\r")
            progress+=1
            if _id not in self.goldstandard:
                self.skipped_queries.append(_id)
                continue
            
            # do not use queries without true positives
            # this add an overhead that can be avoided by refactor the follwing for loop!
            unique_relevants = set(map(lambda x:x["id"], self.goldstandard[_id]))
            if all([ doc["id"] not in unique_relevants for doc in relevance ]):
                self.skipped_queries.append(_id)
                continue
            
            self.sub_set_goldstandard[_id] = defaultdict(list)
            
            for doc in relevance: # for each document that was retrieved
                
                # Splitting and saving the document
                doc_to_sentence = []

                for _itter, position in enumerate(PunktSentenceTokenizer().span_tokenize(doc["text"])):
                    start, end = position 
                    if _itter>0: # fix the start and end position for the abstract
                        start = start-len(doc["title"])-1
                        end = end-len(doc["title"])
                        
                    _doc = {"text":doc["text"][start:end],
                            "start":start,
                            "end":end}
                    
                    doc_to_sentence.append(_doc)
                
                self.collection[doc["id"]] = doc_to_sentence
                
                # goldstandard should store the doc_id and the index of the positive snippets
                if doc["id"] in unique_relevants:
                    _doc_snippets = {
                        "id": doc["id"],
                        "score":doc["score"],
                        "snippet_index": self.find_relevant_snippets(doc_to_sentence, self.goldstandard[_id], len(doc["title"]), doc["id"])
                    }
                    self.sub_set_goldstandard[_id][1].append(_doc_snippets)
                else:
                    _doc_snippets = {
                        "id": doc["id"],
                        "score":doc["score"],
                        "snippet_index": [ 0 for _ in range(len(doc_to_sentence))] # empty relevance
                    }
                    self.sub_set_goldstandard[_id][0].append(_doc_snippets)
                
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
    
    def get_steps(self):

        # an epoch will be defined with respect to the total number of positive pairs
        total_positives = sum(map(lambda x: sum([ len(x[k]) for k in x.keys() if k>0]), self.sub_set_goldstandard.values()))
          
        return total_positives//self.b_size

    def _generate(self, collection=None, **kwargs):
        
        # sanity check
        assert(not(self.sub_set_goldstandard==None and collection==None))
        
        training_data = self.sub_set_goldstandard
        
        # TODO this condition is dependent on the previous
        if collection is None:
            collection = self.collection
            
        while True:
            # TODO check if it is worthit to use numpy to vectorize these operations
            
            y_query = []
            y_pos_doc = []
            y_pos_doc_snippet_label = []
            y_neg_doc = []
            y_neg_doc_snippet_label = []
            
            # build $batch_size triples and yield
            query_indexes = random.sample(population=list(range(len(self.query_list))), k=self.b_size)
            for q_i in query_indexes:
                selected_query = self.query_list[q_i]
                
                # index of the positive documents
                relevance_group = 1
                
                _pos_len = len(training_data[selected_query["id"]][relevance_group])
                pos_doc_index = random.randint(0, _pos_len-1) if _pos_len>1 else 0
                pos_doc_id = training_data[selected_query["id"]][relevance_group][pos_doc_index]
                pos_doc = {"snippets":collection[pos_doc_id["id"]], "score":pos_doc_id["score"]}
                pos_doc_snippet_label = pos_doc_id["snippet_index"]
                
                _neg_len = len(training_data[selected_query["id"]][relevance_group-1])
                neg_doc_index = random.randint(0, _neg_len-1) if _neg_len>1 else 0
                neg_doc_id = training_data[selected_query["id"]][relevance_group-1][neg_doc_index]
                neg_doc = {"snippets":collection[neg_doc_id["id"]], "score":neg_doc_id["score"]}
                neg_doc_snippet_label = neg_doc_id["snippet_index"]
                
                y_query.append(selected_query["query"])
                y_pos_doc.append(pos_doc)
                y_pos_doc_snippet_label.append(pos_doc_snippet_label)
                y_neg_doc.append(neg_doc)
                y_neg_doc_snippet_label.append(neg_doc_snippet_label)
            
            yield (np.array(y_query), np.array(y_pos_doc), np.array(y_pos_doc_snippet_label), np.array(y_neg_doc), np.array(y_neg_doc_snippet_label))
    
    def get_config(self):
        super_config = super().get_config()
        
        data_json = {
            "query_list": self.query_list,
            "goldstandard": self.goldstandard,
            "verbose": self.verbose,
            "sub_set_goldstandard": self.sub_set_goldstandard,
            "collection": self.collection,
        } 
        
        return dict(data_json, **super_config) #fast dict merge


class TestSnippetsCollection(BaseCollection):
    def __init__(self, 
                 query_list,
                 query_docs, 
                 evaluator=None,
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
                       
        query_docs  - previous retrieved method of the retrieved the TOP_K documents
                       {
                           id: [{
                               id: <str>
                               snippets: [<str>, <str>]
                               score: <float>
                           }, ...],
                           ...
                       }
                       
        """
        super(TestSnippetsCollection, self).__init__(**kwargs)
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
    
def grouping_by_q_term(tokenizer, max_sentence_size=30, queries_sw=None, docs_sw=None):
    
    idf_from_id_token = lambda x: math.log(tokenizer.document_count/tokenizer.word_docs[tokenizer.index_word[x]])
    
    def train(data_generator):
        
        while True:
        
            # get the batch triplet
            query, pos_docs, pos_snippet_l, neg_docs, neg_snippet_l= next(data_generator)
            
            # tokenization
            query = tokenizer.texts_to_sequences(query)
            
            if queries_sw is not None:
                for tokenized_query in query:
                    tokenized_query = [token for token in tokenized_query if token not in queries_sw] 
            
            new_pos_docs = []
            new_neg_docs = []
            
            new_pos_docs_snippet_label=[]
            new_neg_docs_snippet_label=[]

            for b in range(len(pos_docs)): # batch itteration
                new_pos_docs.append([])
                new_neg_docs.append([])
                
                new_pos_docs_snippet_label.append([])
                new_neg_docs_snippet_label.append([])

                _temp_pos_docs = map(lambda x:x["text"], pos_docs[b]["snippets"])
                _temp_pos_docs = tokenizer.texts_to_sequences(_temp_pos_docs)

                if docs_sw is not None:
                    for tokenized_docs in _temp_pos_docs:
                        tokenized_docs = [token for token in tokenized_docs if token not in docs_sw] 
                
                # skip batch with empty pos_docs
                if all([ len(sentence)==0  for sentence in _temp_pos_docs]):
                    break # try a new resampling, NOTE THIS IS A EASY FIX PLS REDO THIS!!!!!!!
                             # for obvious reasons
                
                _temp_neg_docs = map(lambda x:x["text"], neg_docs[b]["snippets"])
                _temp_neg_docs = tokenizer.texts_to_sequences(_temp_neg_docs)

                if docs_sw is not None:
                    for tokenized_docs in _temp_neg_docs:
                        tokenized_docs = [token for token in tokenized_docs if token not in docs_sw] 

                # split by exact matching
                for t_q in query[b]:
                    # entry for the query-term
                    new_pos_docs[-1].append([])
                    new_neg_docs[-1].append([])
                    
                    new_pos_docs_snippet_label[-1].append([])
                    new_neg_docs_snippet_label[-1].append([])

                    for snippet_index, pos_sent in enumerate(_temp_pos_docs):
                        # exact math for the pos_document
                        for i,t_pd in enumerate(pos_sent):
                            if t_pd==t_q:
                                new_pos_docs[-1][-1].append(pos_sent)
                                new_pos_docs_snippet_label[-1][-1].append(pos_snippet_l[b][snippet_index])
                                break # no need to check for more than 1 q-term match

                    for snippet_index, neg_sent in enumerate(_temp_neg_docs):
                        for i,t_nd in enumerate(neg_sent):
                            if t_nd==t_q:
                                new_neg_docs[-1][-1].append(neg_sent)
                                new_neg_docs_snippet_label[-1][-1].append(neg_snippet_l[b][snippet_index])
                                break # no need to check for more than 1 q-term match

            if len(new_pos_docs) == len(pos_docs): # if batch is correct    
                yield query, new_pos_docs, new_pos_docs_snippet_label, new_neg_docs, new_neg_docs_snippet_label
    
    def test(data_generator):
        for _id, query, docs in data_generator:

            # tokenization
            tokenized_query = tokenizer.texts_to_sequences([query])[0]
            
            if queries_sw is not None:
                tokenized_query = [token for token in tokenized_query if token not in queries_sw] 
            
            for doc in docs:
                if isinstance(doc["text"], list):
                    continue # cached tokenization

                # sentence splitting
                new_docs = []
                
                _temp_new_docs = []
                doc["offset"] = []
                for start, end in PunktSentenceTokenizer().span_tokenize(doc["text"]):
                    _temp_new_docs.append(doc["text"][start:end])

                    if start<(len(doc["title"])-1):
                        doc["offset"].append(["title",(start, end), doc["text"][start:end], []])
                    else:
                        doc["offset"].append(["abstract", (start-len(doc["title"]), end-len(doc["title"])), doc["text"][start:end], []])

                _temp_new_docs = tokenizer.texts_to_sequences(_temp_new_docs)

                if docs_sw is not None:
                    for tokenized_docs in _temp_new_docs:
                        tokenized_docs = [token for token in tokenized_docs if token not in docs_sw]

                #doc["extra_features"] = compute_extra_features(tokenized_query, _temp_new_docs, idf_from_id_token)+[doc["score"]]

                for k,t_q in enumerate(tokenized_query):
                    new_docs.append([])
                    for l,_new_doc in enumerate(_temp_new_docs):
                        for i,t_d in enumerate(_new_doc):
                            if t_d==t_q:
                                new_docs[-1].append(_new_doc)
                                doc["offset"][l][-1].append(k)
                                break

                                                                    
                doc["text"] = new_docs
                                                                    
            yield _id, tokenized_query, docs
    
    return train, test