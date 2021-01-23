import numpy as np
import random
import tensorflow as tf
import h5py
import pickle
import mmnrm.modelsv2
import math

from datetime import datetime as dt


def set_random_seed(seed_value=42):
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    
def save_model_weights(file_name, model):
    with h5py.File(file_name+".h5", 'w') as f:
        weight = model.get_weights()
        for i in range(len(weight)):
            f.create_dataset('weight'+str(i), data=weight[i])

def load_model_weights(file_name, model):
    with h5py.File(file_name+".h5", 'r') as f:
        weight = []
        for i in range(len(f.keys())):
            weight.append(f['weight'+str(i)][:])
        model.set_weights(weight)

def load_sentence_generator(cfg, tk=None, return_tk=False):

    if tk is None:
        tk = load_tokenizer(cfg)
    
    model_cfg = cfg["model"]
    
    max_input_query = model_cfg["max_q_length"]
    max_input_sentence = model_cfg["max_s_length"]
    max_s_per_q_term = model_cfg["max_s_per_q_term"]
    
    # redundant code... replace
    max_sentences_per_query = model_cfg["max_s_per_q_term"]

    pad_query = lambda x, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                       maxlen=max_input_query,
                                                                                       dtype=dtype, 
                                                                                       padding='post', 
                                                                                       truncating='post', 
                                                                                       value=0)

    pad_sentences = lambda x, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                           maxlen=max_input_sentence,
                                                                                           dtype=dtype, 
                                                                                           padding='post', 
                                                                                           truncating='post', 
                                                                                           value=0)

    pad_docs = lambda x, max_lim, dtype='int32': x[:max_lim] + [[]]*(max_lim-len(x))

    idf_from_id_token = lambda x: math.log(tk.document_count/tk.word_docs[tk.index_word[x]])

    # inline import
    from mmnrm.dataset import sentence_splitter_builderV2
    train_sentence_generator, test_sentence_generator = sentence_splitter_builderV2(tk, 
                                                                                      max_sentence_size=max_input_sentence,
                                                                                      mode=4)
    
    def train_input_generator(data_generator):
        data_generator = train_sentence_generator(data_generator)

        while True:
            query, pos_docs, pos_extra_features, neg_docs, neg_extra_features = next(data_generator)

            query_idf = np.array([list(map(lambda x: idf_from_id_token(x), t_q)) for t_q in query])

            # padding
            for i in range(len(pos_docs)):
                pos_docs[i] = pad_docs(pos_docs[i], max_lim=model_cfg['max_q_length'])
                neg_docs[i] = pad_docs(neg_docs[i], max_lim=model_cfg['max_q_length'])

                for q in range(len(pos_docs[i])):

                    pos_docs[i][q] = pad_docs(pos_docs[i][q], max_lim=model_cfg['max_s_per_q_term'])
                    neg_docs[i][q] = pad_docs(neg_docs[i][q], max_lim=model_cfg['max_s_per_q_term'])

                    pos_docs[i][q] = pad_sentences(pos_docs[i][q])
                    neg_docs[i][q] = pad_sentences(neg_docs[i][q])

            query = pad_query(query)
            query_idf = pad_query(query_idf, dtype="float32")
            
            yield [query, np.array(pos_docs), query_idf], [query,  np.array(neg_docs), query_idf]
            
            
    def test_input_generator(data_generator):

        data_generator = test_sentence_generator(data_generator)

        for ids, queries, l_docs in data_generator:
            
            tokenized_docs = []
            ids_docs = []
            offsets_docs = []
            ids_queries = []
            queries_idf = []
            tokenized_queries = []
            for i in range(len(ids)):
                #tokenization
                query_idf = list(map(lambda x: idf_from_id_token(x), queries[i]))

                for doc in l_docs[i]:

                    padded_doc = pad_docs(doc["text"], max_lim=max_input_query)
                    for q in range(len(padded_doc)):
                        padded_doc[q] = pad_docs(padded_doc[q], max_lim=max_sentences_per_query)
                        padded_doc[q] = pad_sentences(padded_doc[q])
                    tokenized_docs.append(padded_doc)
                    ids_docs.append(doc["id"])
                    offsets_docs.append(doc["offset"])
                    

                # padding
                query = pad_query([queries[i]])[0]
                query = [query] * len(l_docs[i])
                tokenized_queries.append(query)
                
                query_idf = pad_query([query_idf], dtype="float32")[0]
                query_idf = [query_idf] * len(l_docs[i])
                queries_idf.append(query_idf)
                ids_queries.append([ids[i]]*len(l_docs[i]))
                
            yield flat_list(ids_queries), [np.array(flat_list(tokenized_queries)), np.array(tokenized_docs), np.array(flat_list(queries_idf))], ids_docs, offsets_docs
    
    
    if return_tk:
        return train_input_generator, test_input_generator, tk
    else:
        return train_input_generator, test_input_generator
    
    

def load_neural_model(path_to_weights, return_snippets_score=True):
    
    rank_model = load_model(path_to_weights, change_config={"return_snippets_score":return_snippets_score})
    tk = rank_model.tokenizer  

    return rank_model, load_sentence_generator(rank_model.savable_config, tk)
        
def save_model(file_name, model):
    cfg = model.savable_config
    with open(file_name+".cfg","wb") as f:
        pickle.dump(model.savable_config ,f)
        
    # keep using h5py for weights
    save_model_weights(file_name, model)
    
def load_model(file_name, change_config={}):
    
    with open(file_name+".cfg","rb") as f:
        cfg = pickle.load(f)
    
    cfg["model"] = merge_dicts(cfg["model"], change_config)
    
    # create the model with the correct configuration
    model = getattr(mmnrm.modelsv2, cfg['func_name'])(**cfg)
    
    # load weights
    load_model_weights(file_name, model)
    
    return model

def load_model_config(file_name):
    with open(file_name+".cfg","rb") as f:
        cfg = pickle.load(f)
    
    return cfg

def load_tokenizer(cfg):
     
    tk = cfg['tokenizer']['class'].load_from_json(**cfg['tokenizer']['attr'])
    tk.update_min_word_frequency(cfg['tokenizer']['min_freq'])
    
    return tk

def merge_dicts(*list_of_dicts):
    # fast merge according to https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one-in-python
    
    temp = dict(list_of_dicts[0], **list_of_dicts[1])
    
    for i in range(2, len(list_of_dicts)):
        temp.update(list_of_dicts[i])
        
    return temp

def flat_list(x):
    return sum(x, [])

def index_from_list(searchable_list, comparison):
    for i,item in enumerate(searchable_list):
        if comparison(item):
            return i
    return -1

def overlap(snippetA, snippetB):
    """
    snippetA: goldSTD
    """
    if snippetA[0]>snippetB[1] or snippetA[1] < snippetB[0]:
        return 0
    else:
        if snippetA[0]>=snippetB[0] and snippetA[1] <= snippetB[1]:
            return snippetA[1] - snippetA[0] + 1
        if snippetA[0]>=snippetB[0] and snippetA[1] > snippetB[1]:
            return snippetB[1] - snippetA[0] + 1
        if snippetA[0]<snippetB[0] and snippetA[1] <= snippetB[1]:
            return snippetA[1] - snippetB[0] + 1
        if snippetA[0]<snippetB[0] and snippetA[1] > snippetB[1]:
            return snippetB[1] - snippetA[0] + 1
        
    return 0

def to_date(_str):
    for fmt in ("%Y-%m", "%Y-%m-%d", "%Y"):
        try:
            return dt.strptime(_str, fmt)
        except ValueError:
            pass
    raise ValueError("No format found")
