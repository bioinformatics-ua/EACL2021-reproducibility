import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

from mmnrm.layers.interaction import SemanticInteractions, ExactInteractions
from mmnrm.layers.local_relevance import MultipleNgramConvs, MaskedSoftmax, SimpleMultipleNgramConvs
from mmnrm.layers.transformations import *
from mmnrm.layers.aggregation import *


def savable_model(func):
    def function_wrapper(**kwargs):
        
        # create tokenizer
        if 'tokenizer' in kwargs:
            tk = kwargs['tokenizer']['class'].load_from_json(**kwargs['tokenizer']['attr'])
            tk.update_min_word_frequency(kwargs['tokenizer']['min_freq'])
        else:
            raise ValueError('Missing tokenizer parameter in the config')
            
        # create emb matrix
        if 'embedding' in kwargs:
            emb = kwargs['embedding']['class'].maybe_load(**kwargs['embedding']['attr'], tokenizer=tk)
            emb_matrix = emb.embedding_matrix()
            
            # check if it's normalized
            assert(all([ int(np.linalg.norm(v)+0.001)==1 for v in emb_matrix ]))
        else:
            raise ValueError('Missing embedding parameter in the config')
        
        model = func(**kwargs["model"], emb_matrix=emb_matrix)
        kwargs['func_name'] = func.__name__
        model.savable_config = kwargs
        model.tokenizer = tk

        return model

    return function_wrapper


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

@savable_model
def deep_rank(max_q_length=30,
              max_s_per_q_term=5,
              max_s_length=30,
              emb_matrix=None,
              filters=16,
              kernel_size=[3,3],
              aggregation_size=16,
              q_term_weight_mode=0,
              aggregation_mode=0,
              extraction_mode=0,
              score_mode=0,
              return_snippets_score=False,
              train_context_emgeddings=False,
              activation="selu"):
    """
    q_term_weight_mode: 0 - use term aggregation with embeddings
                        1 - use term aggregation with idf
                        
    aggregation_mode: 0 - use Bidirectional GRU
                      1 - use Bidirectional GRU + sig for sentence score follow another Bidirectional GRU for aggregation
                      2 - use Bidirectional GRU + sig for sentence score
                      3 - compute score independently + sig for sentence score
                      4 - mlp + sig
    
    extraction_mode: 0 - use CNN + GlobalMaxPool
                     1 - use CNN + [GlobalMaxPool, GlobalAvgPool]
                     2 - use CNN + [GlobalMaxPool, GlobalAvgPool, GlobalK-maxAvgPool]
                     3 - use CNN + [GlobalMaxPool, GlobalK-maxAvgPool]
                     4 - use CNN + GlobalKmaxPool
                     5 - use MultipleNgramConvs
    
    score_mode: 0 - Dense layer
                1 - MLP
                     
    

    """
    
    if activation=="mish":
        activation = mish
    
    kernel_size = tuple(kernel_size)
    
    return_embeddings = q_term_weight_mode==0
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32") # (None, Q)
    input_doc = tf.keras.layers.Input((max_q_length, max_s_per_q_term, max_s_length), dtype="int32") # (None, P, S)
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    
    interactions = SemanticInteractions(emb_matrix, return_embeddings=return_embeddings, learn_context=train_context_emgeddings)
    
    if extraction_mode==0:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), activation=activation)
        pool = tf.keras.layers.GlobalMaxPool2D()

        def extract(x):
            if return_embeddings:
                x, query_embeddings, _ = interactions(x)
            else:
                x = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            x = conv(x)
            x = pool(x)
            return x, query_embeddings
        
    elif extraction_mode in [1, 2, 3]:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="SAME", activation=activation)
        max_pool = tf.keras.layers.GlobalMaxPool2D()
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        masked_avg_pool = GlobalMaskedAvgPooling2D()
        kmax_avg_pool = GlobalKmaxAvgPooling2D(kmax=5)
        concatenate = tf.keras.layers.Concatenate(axis=-1)
        
        def extract(x):
            if return_embeddings:
                x_interaction, query_embeddings, _ = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            x = conv(x_interaction)
            max_x = max_pool(x)
            _concat = [max_x]
            if extraction_mode in [1, 2]:
                avg_x = avg_pool(x)
                _concat.append(avg_x)
            if extraction_mode in [2, 3]:
                kmax_x = kmax_avg_pool(x)
                _concat.append(kmax_x)
            x = concatenate(_concat)
            
            return x, query_embeddings
    elif extraction_mode==4:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="SAME", activation=activation)
        kmax_pool = GlobalKmax2D()
        
        def extract(x):
            if return_embeddings:
                x_interaction, query_embeddings, _ = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            x = conv(x_interaction)
            x = kmax_pool(x)

            return x, query_embeddings
    elif extraction_mode==5:
        
        ngram_convs = MultipleNgramConvs(max_ngram=3, k_max=2,filters=filters, k_polling_avg = None, activation=activation)
        gru_pacrr = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation=activation))
        squeeze_l = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        def extract(x):
            if return_embeddings:
                x_interaction, query_embeddings, _ = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
                
            x = ngram_convs(x_interaction)
            x = gru_pacrr(x)
            x = squeeze_l(x)
            
            return x, query_embeddings
        
    elif extraction_mode==6:   
        ngrams_convs = SimpleMultipleNgramConvs(max_ngram=3, filters=filters, activation=activation)
        
        max_pool = tf.keras.layers.GlobalMaxPool2D()
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        kmax_avg_pool = GlobalKmaxAvgPooling2D(kmax=5)
        concatenate = tf.keras.layers.Concatenate(axis=-1)
        
        def extract(x):
            if return_embeddings:
                x_interaction, query_embeddings, _ = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            
            ngrams_signals_x = []
            for t_x in ngrams_convs(x_interaction):
                _max = max_pool(t_x)
                _avg = avg_pool(t_x)
                _kavg = kmax_avg_pool(t_x)
                ngrams_signals_x.append(concatenate([_max, _avg, _kavg]))
            
            x = concatenate(ngrams_signals_x)
            
            return x, query_embeddings
        
    elif extraction_mode==7:  
        conv_sem = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="SAME", activation=activation)
        conv_exact = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="SAME", activation=activation)
        max_pool = tf.keras.layers.GlobalMaxPool2D()
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        kmax_avg_pool = GlobalKmaxAvgPooling2D(kmax=5)
        concatenate = tf.keras.layers.Concatenate(axis=-1)
        
        exact_signals = ExactInteractions()
        
        def extract(x):
            if return_embeddings:
                x_interaction, query_embeddings, _ = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            
            x_exact = exact_signals(x)
            
            x = [conv_sem(x_interaction), conv_exact(x_exact)]
            
            x = [ concatenate([max_pool(_x), avg_pool(_x), kmax_avg_pool(_x)]) for _x in x]
            
            x = concatenate(x)

            return x, query_embeddings
        
    elif extraction_mode==8:  
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="SAME", activation=activation)

        max_pool = tf.keras.layers.GlobalMaxPool2D()
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        kmax_avg_pool = GlobalKmaxAvgPooling2D(kmax=5)
        concatenate = tf.keras.layers.Concatenate(axis=-1)
        
        exact_signals = ExactInteractions()
        
        def extract(x):
            if return_embeddings:
                x_interaction, query_embeddings, _ = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            
            x_exact = exact_signals(x)
            x = concatenate([x_interaction, x_exact])
            x = conv(x)
            
            x = concatenate([max_pool(x), avg_pool(x), kmax_avg_pool(x)])

            return x, query_embeddings
        
        
    else:
        raise RuntimeError("invalid extraction_mode")
        
    if aggregation_mode==0:
        aggregation_senteces = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(aggregation_size))
        
    elif aggregation_mode==1:

        l1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1, return_sequences=True), merge_mode="sum")
        l2 = tf.keras.layers.Activation('sigmoid')
        l3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(aggregation_size), merge_mode="sum")

        def aggregation_senteces(x):
            x = l1(x)
            x = l2(x)
            x = l3(x)

            return x
        
    elif aggregation_mode==2:
        
        l1_a = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1, return_sequences=True), merge_mode="sum")
        l2_a = tf.keras.layers.Activation('sigmoid')
        l3_a = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        
        def aggregation_senteces(x):
            x = l1_a(x)
            x = l2_a(x)
            x = l3_a(x)
            return x
    elif aggregation_mode==3:
        l1_a = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="sigmoid"))
        l2_a = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        
        def aggregation_senteces(x):
            x = l1_a(x)
            x = l2_a(x)
            return x    
    elif aggregation_mode==4:
        
        mlp = tf.keras.models.Sequential()
        mlp.add(tf.keras.layers.Dense(aggregation_size, activation=activation))
        mlp.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        l1_a = tf.keras.layers.TimeDistributed(mlp)
        l2_a = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        
        def aggregation_senteces(x):
            x = l1_a(x)
            x = l2_a(x)
            return x  
    elif aggregation_mode==5:
        l1_a = SelfAttention(aggregation=False)
        l2_a = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="sigmoid"))
        l3_a = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        
        def aggregation_senteces(x):
            x = l1_a(x)
            x = l2_a(x)
            x = l3_a(x)
            return x    
    else:
        raise RuntimeError("invalid aggregation_mode")
        
    aggregation = TermAggregation(aggregate=False)
    aggregation_sum = tf.keras.layers.Lambda(lambda x:K.sum(x, axis=1))
    
    if score_mode==0:
        output_score = tf.keras.layers.Dense(1)
    elif score_mode==1:
        l1_score = tf.keras.layers.Dense(max_q_length, activation=activation)
        l2_score = tf.keras.layers.Dense(1)
        
        def output_score(x):
            x = l1_score(x)
            x = l2_score(x)
            
            return x
    else:
        raise RuntimeError("invalid score_mode")
    
    input_doc_unstack = tf.unstack(input_doc, axis=1)
    
    output_i = []
    for input_i in input_doc_unstack:
        input_i_unstack = tf.unstack(input_i, axis=1) 
        
        output_j = []
        for input_j in input_i_unstack:
            _out, query_embeddings = extract([input_query, input_j])
            output_j.append(_out) # [None, FM]
        output_j_stack = tf.stack(output_j, axis=1) # [None, P_Q, FM]
        
        output_i.append(aggregation_senteces(output_j_stack)) # [None, FM]
        
    output_i_stack = tf.stack(output_i, axis=1)  # [None, Q, FM]
    
    # aggregation
    q_term_weight_vector = aggregation([output_i_stack, query_embeddings])
    doc_vector = aggregation_sum(q_term_weight_vector)
    
    # score
    score = output_score(doc_vector)
    
    _output = [score]
    if return_snippets_score:
        _output.append(q_term_weight_vector)
        
    model = tf.keras.models.Model(inputs=[input_query, input_doc, input_query_idf], outputs=_output)
    model.sentence_tensor = output_i_stack
    return model
