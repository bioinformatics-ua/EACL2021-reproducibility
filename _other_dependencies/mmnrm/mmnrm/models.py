import tensorflow as tf
from tensorflow.keras import backend as K

from mmnrm.layers.interaction import SemanticInteractions, ExactInteractions
from mmnrm.layers.local_relevance import MultipleNgramConvs, MaskedSoftmax
from mmnrm.layers.transformations import *
from mmnrm.layers.aggregation import *

def build_PACRR(max_q_length,
                max_d_length,
                emb_matrix = None,
                learn_context = False,
                trainable_embeddings = False,
                learn_term_weights = False,
                dense_hidden_units = None,
                max_ngram = 3,
                k_max = 2,
                activation="relu",
                out_put_dim = 1,
                return_embeddings=False,
                shuffle_query_terms = False, 
                k_polling_avg = None, # do k_polling avg after convolution
                polling_avg = False, # do avg polling after convolution
                use_mask = True,
                filters=32, # can be a list or a function of input features and n-gram
                name_model = None,
                **kwargs):
    
    prefix_name = ""
    
    # init layers
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32")
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    input_sentence = tf.keras.layers.Input((max_d_length,), dtype="int32")
    
    if emb_matrix is None:
        interaction = ExactInteractions()
    else:
        interaction = SemanticInteractions(emb_matrix, 
                                       learn_term_weights=learn_term_weights, 
                                       trainable_embeddings=trainable_embeddings,
                                       learn_context=learn_context,
                                       use_mask=True,
                                       return_embeddings=return_embeddings)
        
    ngram_convs = MultipleNgramConvs(max_ngram=max_ngram,
                                     k_max=k_max,
                                     k_polling_avg=k_polling_avg,
                                     polling_avg=polling_avg,
                                     use_mask=use_mask,
                                     filters=filters,
                                     activation=activation)
    softmax_IDF = MaskedSoftmax()
    
    if use_mask:
        concatenate = MaskedConcatenate(0)
    else:
        concatenate = tf.keras.layers.Concatenate()
    
    if dense_hidden_units is None:
        aggregation_layer = tf.keras.layers.LSTM(out_put_dim, 
                                    dropout=0.0, 
                                    recurrent_regularizer=None, 
                                    recurrent_dropout=0.0, 
                                    unit_forget_bias=True, 
                                    recurrent_activation="hard_sigmoid", 
                                    bias_regularizer=None, 
                                    activation=activation, 
                                    recurrent_initializer="orthogonal", 
                                    kernel_regularizer=None, 
                                    kernel_initializer="glorot_uniform",
                                    unroll=True) # speed UP!!!
    elif isinstance(dense_hidden_units, list) :
        def _network(x):
            x = tf.keras.layers.Flatten()(x)
            for i,h in enumerate(dense_hidden_units):
                x = tf.keras.layers.Dense(h, activation="relu", name="aggregation_dense_"+str(i))(x)
            dout = tf.keras.layers.Dense(1, name="aggregation_output")(x)
            return dout
        
        aggregation_layer = _network
    else:
        raise RuntimeError("dense_hidden_units must be a list with the hidden size per layer")
    
    # build layers
    
    norm_idf = K.expand_dims(softmax_IDF(input_query_idf))
    
    if return_embeddings:
        _out = interaction([input_query, input_sentence])
        x = _out[0]
        embeddings = _out[1]
    else:
        x = interaction([input_query, input_sentence])
        
      
      
    x = ngram_convs(x)
    x = concatenate([x, norm_idf])
    if shuffle_query_terms:
        shuffle = ShuffleRows()
        prefix_name += "S"
        x = shuffle(x)
    x = aggregation_layer(x)
    
    
    if name_model is None:
        name_model = (prefix_name+"_" if prefix_name != "" else "") + "PACRR"
    
    if return_embeddings:
        return tf.keras.models.Model(inputs=[input_query, input_sentence, input_query_idf], outputs=[x, embeddings])
    else:
        return tf.keras.models.Model(inputs=[input_query, input_sentence, input_query_idf], outputs=x)

def sentence_PACRR(pacrr, sentence_per_doc, type_combination=0, activation="relu"):
    """
    type_combination - 0: use MLP
                       1: use WeightedCombination + MLP
                       2: GRU
    """
    max_q_length = pacrr.input[0].shape[1]
    max_d_length = pacrr.input[1].shape[1]
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32") # (None, Q)
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32") # (None, Q)
    input_doc = tf.keras.layers.Input((sentence_per_doc, max_d_length), dtype="int32") # (None, P, S)
    
    #aggregate = tf.keras.layers.GRU(1, activation="relu")
    #aggregate = WeightedCombination()
    
    def aggregate(x):
        #x = tf.keras.layers.Dense(25, activation="relu")(x)
        x = KmaxAggregation(k=5)(x)
        #x = tf.squeeze(x, axis=-1)
        x = tf.keras.layers.Dense(6, activation="selu")(x)
        return tf.keras.layers.Dense(1, activation=None)(x)
    
    #def aggregate(x):
        #x = tf.keras.layers.Dense(25, activation="relu")(x)
    #    return K.max(tf.squeeze(x, axis=-1), axis=-1, keepdims=True)
    
    sentences = tf.unstack(input_doc, axis=1) #[(None,S), (None,S), ..., (None,S)]
    pacrr_sentences = []
    for sentence in sentences:
        pacrr_sentences.append(pacrr([input_query, sentence, input_query_idf]))
        
    pacrr_sentences = tf.stack(pacrr_sentences, axis=1)
    #pacrr_sentences = tf.squeeze(pacrr_sentences, axis=-1)
    score = aggregate(pacrr_sentences)
    
    return tf.keras.models.Model(inputs=[input_query, input_doc, input_query_idf], outputs=score)
        
def semantic_exact_PACRR(semantic_pacrr_args,
                         exact_pacrr_args,
                         type_combination=0,
                         semantic_filter_threshold=None,
                         dense_hidden_units=[4]):
    
    """
    type_combination - 0: use MLP
                       1: use WeightedCombination + MLP
                       2: use Sum over score
                       3: use self-attention over query + dense + attend
                       4: use RareWordFreqCombine
    """
    
    assert(semantic_pacrr_args["max_q_length"]==exact_pacrr_args["max_q_length"])
    assert(semantic_pacrr_args["max_d_length"]==exact_pacrr_args["max_d_length"])
    
    return_embeddings = type_combination in [3]

    max_q_length = semantic_pacrr_args["max_q_length"]
    max_d_length = semantic_pacrr_args["max_d_length"]
    
    # init layers
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32")
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    input_sentence = tf.keras.layers.Input((max_d_length,), dtype="int32")
    
    # build
    semantic_pacrr = build_PACRR(**semantic_pacrr_args)
    exact_pacrr = build_PACRR(**exact_pacrr_args)

    def _aggregate(x):
        if type_combination==0:
            return tf.keras.layers.Concatenate(axis=-1)(x)
        elif type_combination==1:
            x = tf.keras.layers.Lambda(lambda x: K.concatenate(list(map(lambda y: K.expand_dims(y), x))) )(x)
            return WeightedCombination()(x)
        elif type_combination==2:
            x = tf.keras.layers.Concatenate(axis=-1)(x)
            return K.sum(x, axis=-1, keepdims=True)
        elif type_combination==3:
            query_attn = SelfAttention()(embeddings, mask=_mask)
            score_query = tf.keras.layers.Dense(1, activation="sigmoid")(query_attn)
            return x[0]*score_query + (1-score_query)*x[1]
        elif type_combination==4:
            return RareWordFreqCombine(semantic_filter_threshold)([x[1], x[0], input_query])
        else:
            raise RuntimeError("invalid type_combination")
        
    
    def _score(x):
        if type_combination in [2,3,4]:
            return x # identity
        
        for i,h in enumerate(dense_hidden_units):
            x = tf.keras.layers.Dense(h, activation="relu")(x)
        return tf.keras.layers.Dense(1, activation="relu")(x)
    # build layers
    
    if semantic_filter_threshold is not None:
        semantic_filter_mask = ReplaceValuesByThreashold(semantic_filter_threshold, return_filter_mask=True)
        semantic_filter = ReplaceValuesByThreashold(semantic_filter_threshold, return_filter_mask=False)
        semantic_filter_idf = ReplaceValuesByMask()
        
        input_query_filter, _mask = semantic_filter_mask(input_query)
        input_sentence_filter = semantic_filter(input_sentence)
        input_query_idf_filter = semantic_filter_idf([input_query_idf, _mask])
    
    # semantic pacrr with embeddings
    if return_embeddings:
        semantic_repr, embeddings = semantic_pacrr([input_query_filter, input_sentence_filter, input_query_idf_filter])
    else:
        semantic_repr = semantic_pacrr([input_query_filter, input_sentence_filter, input_query_idf_filter])
        
    exact_repr = exact_pacrr([input_query, input_sentence, input_query_idf])
    
    combined = _aggregate([semantic_repr, exact_repr])
    
    score = _score(combined)
    
    return tf.keras.models.Model(inputs=[input_query, input_sentence, input_query_idf], outputs=score)

def experimental_semantic_exact_PACRR(semantic_pacrr_args,
                                      exact_pacrr_args,
                                      semantic_filter_threshold=None,
                                      filters=16,
                                      activation="relu"):
    
    assert(semantic_pacrr_args["max_q_length"]==exact_pacrr_args["max_q_length"])
    assert(semantic_pacrr_args["max_d_length"]==exact_pacrr_args["max_d_length"])
    
    max_q_length = semantic_pacrr_args["max_q_length"]
    max_d_length = semantic_pacrr_args["max_d_length"]
    
    # init layers
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32")
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    input_sentence = tf.keras.layers.Input((max_d_length,), dtype="int32")
    
    # exact pacrr
    e_interaction = ExactInteractions()
    
    e_ngram_convs = MultipleNgramConvs(filters=filters, 
                                       activation=activation,
                                       max_ngram=3,
                                       k_max=2,
                                       k_polling_avg=None,
                                       polling_avg=False)
    e_softmax_IDF = MaskedSoftmax()

    e_concatenate = MaskedConcatenate(0)

    e_aggregation_layer = tf.keras.layers.LSTM(1, 
                                    dropout=0.0, 
                                    recurrent_regularizer=None, 
                                    recurrent_dropout=0.0, 
                                    unit_forget_bias=True, 
                                    recurrent_activation="hard_sigmoid", 
                                    bias_regularizer=None, 
                                    activation=activation, 
                                    recurrent_initializer="orthogonal", 
                                    kernel_regularizer=None, 
                                    kernel_initializer="glorot_uniform",
                                    unroll=True) # speed UP!!!
    
    # build layers
    
    e_norm_idf = K.expand_dims(e_softmax_IDF(input_query_idf))
    
    e_out_interaction = e_interaction([input_query, input_sentence])

    e_x = e_ngram_convs(e_out_interaction)
    e_x = e_concatenate([e_x, e_norm_idf])

    e_x = e_aggregation_layer(e_x)
    
    # semantic pacrr
    
    semantic_filter_mask = ReplaceValuesByThreashold(semantic_filter_threshold, return_filter_mask=True)
    semantic_filter = ReplaceValuesByThreashold(semantic_filter_threshold, return_filter_mask=False)
    semantic_filter_idf = ReplaceValuesByMask()

    s_interaction = SemanticInteractions(semantic_pacrr_args["emb_matrix"])
    s_combined_interaction = tf.keras.layers.Lambda(lambda x: x[0]-x[1], mask=lambda x,m:m[0])
        
    s_ngram_convs = MultipleNgramConvs(filters=filters, 
                                       activation=activation,
                                       max_ngram=3,
                                       k_max=2,
                                       k_polling_avg=None,
                                       polling_avg=False)
    s_softmax_IDF = MaskedSoftmax()

    s_concatenate = MaskedConcatenate(0)

    s_aggregation_layer = tf.keras.layers.LSTM(1, 
                                    dropout=0.0, 
                                    recurrent_regularizer=None, 
                                    recurrent_dropout=0.0, 
                                    unit_forget_bias=True, 
                                    recurrent_activation="hard_sigmoid", 
                                    bias_regularizer=None, 
                                    activation=activation, 
                                    recurrent_initializer="orthogonal", 
                                    kernel_regularizer=None, 
                                    kernel_initializer="glorot_uniform",
                                    unroll=True) # speed UP!!!
    
    # build layers
    input_query_filter, _mask = semantic_filter_mask(input_query)
    input_sentence_filter = semantic_filter(input_sentence)
    input_query_idf_filter = semantic_filter_idf([input_query_idf, _mask])
    
    s_out_interaction = s_interaction([input_query_filter, input_sentence_filter]) 
    
    s_out_interaction = s_combined_interaction([s_out_interaction, e_out_interaction])
   
    s_norm_idf = K.expand_dims(s_softmax_IDF(input_query_idf_filter))

    s_x = s_ngram_convs(s_out_interaction)
    s_x = s_concatenate([s_x, s_norm_idf])

    s_x = s_aggregation_layer(s_x)
    
    ## Agregation
    score = RareWordFreqCombine(semantic_filter_threshold)([s_x, e_x, input_query])
    
    return tf.keras.models.Model(inputs=[input_query, input_sentence, input_query_idf], outputs=score)
 
def simple_sentence_match(max_q_length,
                          max_s_length,
                          max_s_per_doc,
                          emb_matrix,
                          learn_context = False,
                          trainable_embeddings = False,
                          learn_term_weights = False,
                          use_mask=True,
                          matching_extraction_mode=0,
                          q_terms_aggregation = 0,
                          hidden_term_aggregation = 6,
                          max_ngram = 3,
                          k_max = 2,
                          filters = 16,
                          activation="relu"):
    """
    q_terms_aggregation: 0 - bidirectional lstm
                         1 - self-attention
                         
    matching_extraction_mode: 0 - multiple convs with k_max 2
                              1 - conv with global max pooling
    """
    
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32") # (None, Q)
    input_doc = tf.keras.layers.Input((max_s_per_doc, max_s_length), dtype="int32") # (None, P, S)
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    
    softmax_IDF = MaskedSoftmax()
    
    concatenate = MaskedConcatenate(0)
    
    interaction = SemanticInteractions(emb_matrix, 
                                       learn_term_weights=learn_term_weights, 
                                       trainable_embeddings=trainable_embeddings,
                                       learn_context=learn_context,
                                       use_mask=use_mask,
                                       return_embeddings=True)
    
    # convolutions     
    ngram_convs = MultipleNgramConvs(max_ngram=max_ngram,
                                     k_max=k_max,
                                     k_polling_avg=None,
                                     polling_avg=False,
                                     use_mask=use_mask,
                                     filters=filters,
                                     activation=activation)
    
    
    if q_terms_aggregation==0:
        sentence_signal = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_term_aggregation, activation=activation))
    elif q_terms_aggregation==1:
        sentence_signal = SelfAttention(attention_dimension=hidden_term_aggregation)
    else:
        raise KeyValueError("invalid value for q_terms_aggregation")
    
    
    def aggregate(x):
        
        x = KmaxAggregation(k=2)(x)
        # x = tf.squeeze(x, axis=-1)
        x = tf.keras.layers.Dense(6, activation=activation)(x)
        return tf.keras.layers.Dense(1, activation=None)(x)
    
    
    
    input_sentences = tf.unstack(input_doc, axis=1) # [(None,S), (None,S), ..., (None,S)]
    sentences_hidden = []
    
    for input_sentence in input_sentences:
        _out = interaction([input_query, input_sentence])
        x = _out[0]
        query_embedding = _out[1]
        
        x = ngram_convs(x)
    
        norm_idf = K.expand_dims(softmax_IDF(input_query_idf))
    
        x = concatenate([x, norm_idf])
    
        x = sentence_signal(x)
        
        sentences_hidden.append(x)
        
    sentences_hidden = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(sentences_hidden)
    
    combined = aggregate(sentences_hidden)
    
    return tf.keras.models.Model(inputs=[input_query, input_doc, input_query_idf], outputs=combined)
    
    
def deep_rank(max_q_length,
              max_s_length,
              max_s_per_q_term,
              emb_matrix,
              filters=16,
              gru=16,
              q_term_weight_mode=0,
              aggregation_mode=0,
              extraction_mode=0):
    """
    q_term_weight_mode: 0 - use term aggregation with embeddings
                        1 - use term aggregation with idf
                        
    aggregation_mode: 0 - use Bidirectional GRU
                      1 - use Bidirectional GRU + sig for sentence score follow another Bidirectional GRU for aggregation
                      2 - use Bidirectional GRU + sig for sentence score
                      3 - compute score independently + sig for sentence score
    
    extraction_mode: 0 - use CNN + GlobalMaxPool
                     1 - use CNN + [GlobalMaxPool, GlobalAvgPool]
                     2 - use CNN + [GlobalMaxPool, GlobalAvgPool, GlobalK-maxAvgPool]
                     3 - use CNN + [GlobalMaxPool, GlobalK-maxAvgPool]
                     4 - use CNN + GlobalKmaxPool

    """
    
    initialized_vars = locals()
    
    return_embeddings = q_term_weight_mode==0
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32") # (None, Q)
    input_doc = tf.keras.layers.Input((max_q_length, max_s_per_q_term, max_s_length), dtype="int32") # (None, P, S)
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    
    interactions = SemanticInteractions(emb_matrix, return_embeddings=return_embeddings)
    
    if extraction_mode==0:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), activation="selu")
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
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3),padding="SAME", activation="selu")
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
            elif extraction_mode in [2, 3]:
                kmax_x = kmax_avg_pool(x)
                _concat.append(kmax_x)
            x = concatenate(_concat)
            
            return x, query_embeddings
    elif extraction_mode==4:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3),padding="SAME", activation="selu")
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
        
    else:
        raise RuntimeError("invalid extraction_mode")
        
    if aggregation_mode==0:
        aggregation_senteces = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru))
        
    elif aggregation_mode==1:

        l1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1, return_sequences=True), merge_mode="sum")
        l2 = tf.keras.layers.Activation('sigmoid')
        l3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru), merge_mode="sum")

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
    else:
        raise RuntimeError("invalid aggregation_mode")
        
    aggregation = TermAggregation()
    
    output_score = tf.keras.layers.Dense(1)
    
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
    doc_vector = aggregation([output_i_stack, query_embeddings])
    
    # score
    score = output_score(doc_vector)
    
    return tf.keras.models.Model(inputs=[input_query, input_doc, input_query_idf], outputs=score), output_i_stack
        


def deep_snippet_ranker(max_q_length,
                        max_s_length,
                        max_s_per_doc,
                        emb_matrix,
                        filters=16,
                        gru=16,
                        conditional_speed_up=False,
                        q_term_weight_mode=0,
                        aggregation_mode=0,
                        score_mode=0,
                        extract_mode=0):
    """
    q_term_weight_mode: 0 - use idf
                        1 - use softmax(idf)
                        2 - use softmax(embedings)
    
    aggregation_mode: 0 - use Bidirectional GRU + sig for sentence score follow another Bidirectional GRU for aggregation
                      1 - use Bidirectional GRU + sig for sentence score follow GRU to score
    
    extract_mode: 0 - Convolution + global max pool
                  1 - MultipleNgramConvs + idf + flat
    
    score_mode: 0 - dense
                1 - linear
                2 - mlp
    """
    return_embeddings=False
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32") # (None, Q)
    input_doc = tf.keras.layers.Input((max_s_per_doc, max_s_length), dtype="int32") # (None, P, S)
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    
    
    #softmax_IDF(x[1])
    
    if q_term_weight_mode==0:
        def q_term_weight_fn(x):
            interaction_weighted = tf.squeeze(x[0], axis=-1)*K.expand_dims(x[1], axis=-1)
            return K.expand_dims(interaction_weighted, axis=-1)
        
    elif q_term_weight_mode==1:
        softmax_IDF = MaskedSoftmax()
        
        def q_term_weight_fn(x):
            interaction_weighted = tf.squeeze(x[0], axis=-1)*K.expand_dims(softmax_IDF(x[1]), axis=-1)
            return K.expand_dims(interaction_weighted, axis=-1)
        
    elif q_term_weight_mode==2:
        return_embeddings=True
        q_term_aggregation = TermAggregation(aggregate=False)
        
        def q_term_weight_fn(x):
            interaction_m = tf.squeeze(x[0], axis=-1)
            return  K.expand_dims(q_term_aggregation([interaction_m, x[1]]), axis=-1)
    else:
        raise RuntimeError("invalid q_term_weight_mode")

    
    
    if extract_mode==0:
        interactions = SemanticInteractions(emb_matrix, return_embeddings=return_embeddings)
        softmax_IDF = MaskedSoftmax()
        normalize_interactions_idf = tf.keras.layers.Lambda(q_term_weight_fn, mask=lambda x,mask=None: x[0])
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3))
        pool = tf.keras.layers.GlobalMaxPool2D()

        def extract(x):
            if return_embeddings:
                out, q_embedding, _ = interactions([x[0], x[1]])
                x[2] = q_embedding
            else:
                out = interactions([x[0], x[1]])
            x = normalize_interactions_idf([out, x[2]])
            x = conv(x)
            x = pool(x)
            return x
    
    elif extract_mode==1:
        
        interactions = SemanticInteractions(emb_matrix, return_embeddings=return_embeddings)
        
        ngrams_conv = MultipleNgramConvs(3,
                                           2,
                                           k_polling_avg = None, # do k_polling avg after convolution
                                           polling_avg = False, # do avg polling after convolution
                                           use_mask = True,
                                           filters=filters, # can be a list or a function of input features and n-gram
                                           activation="selu")
        
        softmax_IDF = MaskedSoftmax()
        m_concatenate = MaskedConcatenate(0)
        flat = tf.keras.layers.Flatten()
        
        def extract(x):
            if return_embeddings:
                raise RuntimeError("extract_mode 1 does not support")
            else:
                out = interactions([x[0], x[1]])
            idf_norm = K.expand_dims(softmax_IDF(x[2]))
            x = ngrams_conv(out)
            x = m_concatenate([idf_norm, x])
            x = flat(x)
            return x
    elif extract_mode==2:
        interactions = SemanticInteractions(emb_matrix, return_embeddings=return_embeddings)
        normalize_interactions_idf = tf.keras.layers.Lambda(q_term_weight_fn, mask=lambda x,mask=None: x[0])
        conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3))
        pool_1 = tf.keras.layers.MaxPool2D()
        conv_2 = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=(3,3))
        pool_2 = tf.keras.layers.MaxPool2D()
        pool_3 = tf.keras.layers.MaxPool2D()
        flatten_3 = tf.keras.layers.Flatten()
        def extract(x):
            if return_embeddings:
                out, q_embedding, _ = interactions([x[0], x[1]])
                x[2] = q_embedding
            else:
                out = interactions([x[0], x[1]])
            x = normalize_interactions_idf([out, x[2]])
            x = conv_1(x)
            x = pool_1(x)
            x = conv_2(x)
            x = pool_2(x)
            x = pool_3(x)
            x = flatten_3(x)
            return x
    else:
        raise RuntimeError("invalid extract_mode")
    
    if aggregation_mode==0:
        
        l1_a = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1, return_sequences=True), merge_mode="sum")
        l2_a = tf.keras.layers.Activation('sigmoid')
        l3_a = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru), merge_mode="sum")
        
        def aggregation_senteces(x):
            x = l1_a(x)
            x = l2_a(x)
            x = l3_a(x)
            return x # tf.keras.layers.Activation('relu')(x)
        
    elif aggregation_mode==1:
        
        l1_a = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1, return_sequences=True), merge_mode="sum")
        l2_a = tf.keras.layers.Activation('selu')
        l3_a = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        
        def aggregation_senteces(x):
            x = l1_a(x)
            x = l2_a(x)
            x = l3_a(x)
            return x
        
    else:
        raise RuntimeError("invalid aggregation_mode")
        
    if score_mode==0:
        output_score = tf.keras.layers.Dense(1)
    elif score_mode==1:
        output_score = lambda x:x # identity
    elif score_mode==2:
        l1_s = tf.keras.layers.Dense(max_s_per_doc, activation="selu")
        l2_s = tf.keras.layers.Dense(1, activation="selu")
        
        def output_score(x):
            x = l1_s(x)
            x = l2_s(x)
            return x
    else:
        raise RuntimeError("invalid score_mode")
        
    input_sentences = tf.unstack(input_doc, axis=1)
    sentences_features = []
    
    for input_sentence in input_sentences: 
        # (None, S)
        sentences_features.append(extract([input_query, input_sentence, input_query_idf]))
    
    sentences_features_stack = tf.stack(sentences_features, axis=1)
    
    document_dense = aggregation_senteces(sentences_features_stack)
    #print(document_dense)
    score = output_score(document_dense)
    
    return tf.keras.models.Model(inputs=[input_query, input_doc, input_query_idf], outputs=score)


def q_aware_sentence_ranker(max_q_length,
                            max_s_length,
                            max_s_per_doc,
                            emb_matrix,
                            filters=16):

    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32") # (None, Q)
    input_doc = tf.keras.layers.Input((max_s_per_doc, max_s_length), dtype="int32") # (None, P, S)
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    
    semantic_interactions = SemanticInteractions(emb_matrix)
    exact_interactions = ExactInteractions()
    softmax_IDF = MaskedSoftmax()
    semantic_conv_layer = tf.keras.layers.Conv2D(filters, 
                                         (3,3),
                                         activation="tanh",
                                         padding="SAME",
                                         dtype="float32")
    
    exact_conv_layer = tf.keras.layers.Conv2D(filters, 
                                         (3,3),
                                         activation="tanh",
                                         padding="SAME",
                                         dtype="float32")
    
   
    
    channels_pool_layer = tf.keras.layers.Lambda(lambda x:K.max(x, axis=-1), name="channels_pool_layer")
    squeeze_layer = tf.keras.layers.Lambda(lambda x:tf.squeeze(x, axis=-1), name="squeeze_pool_layer")
    concatenate_layer = tf.keras.layers.Concatenate()
    
    max_by_row_layer = tf.keras.layers.Lambda(lambda x:K.max(x, axis=-1, keepdims=True), name="max_by_row_layer")
    avg_by_row_layer = tf.keras.layers.Lambda(lambda x:K.mean(x, axis=-1, keepdims=True), name="avg_by_row_layer")

    s_l1 = tf.keras.layers.Dense(8, activation="selu", name="mlp_l1")
    s_l2 = tf.keras.layers.Dense(1, name="mlp_l2")
    
    def mlp_sentences(x):
        x = s_l1(x)
        x = s_l2(x)
        return x
    
    def setence_model(x):
        # connections
        semantic_matrix = semantic_interactions([x[0], x[1]])
        exact_matrix = exact_interactions([x[0], x[1]])
        # print(semantic_matrix._keras_mask)
        semantic_feature_maps = semantic_conv_layer(semantic_matrix)
        semantic_feature_map = channels_pool_layer(semantic_feature_maps)
        semantic_feature_map_max = max_by_row_layer(semantic_feature_map)
        semantic_feature_map_avg = avg_by_row_layer(semantic_feature_map)
        
        exact_feature_maps = exact_conv_layer(exact_matrix)
        exact_feature_map = channels_pool_layer(exact_feature_maps)
        exact_feature_map_max = max_by_row_layer(exact_feature_map)
        exact_feature_map_avg = avg_by_row_layer(exact_feature_map)

        semantic_matrix = squeeze_layer(semantic_matrix)
        semantic_max = max_by_row_layer(semantic_matrix)
        semantic_avg = avg_by_row_layer(semantic_matrix)
        
        exact_matrix = squeeze_layer(exact_matrix)
        exact_max = max_by_row_layer(exact_matrix)
        exact_avg = avg_by_row_layer(exact_matrix)

        
        
        features_concat = concatenate_layer([semantic_max, semantic_avg, semantic_feature_map_max, semantic_feature_map_avg, 
                                                      exact_max, exact_avg, exact_feature_map_max, exact_feature_map_avg])
        
        # custom layer to aplly the mlp and do the masking stuff
        features_by_q_term = tf.unstack(features_concat, axis=1)
        features_q_term_score = []
        for f_q_term in features_by_q_term:
            features_q_term_score.append(mlp_sentences(f_q_term))
        score_by_q_term = tf.stack(features_q_term_score, axis=1)
        
        # compute idf importance
        idf_importance = K.expand_dims(softmax_IDF(x[2]))

        return tf.squeeze(tf.linalg.matmul(score_by_q_term, idf_importance, transpose_a=True), axis=-1)
    
    # normal flow model
    input_sentences = tf.unstack(input_doc, axis=1)
    sentences_features = []
    
    for input_sentence in input_sentences: 
        # (None, S)
        sentences_features.append(setence_model([input_query, input_sentence, input_query_idf]))
    
    sentences_features_stack = tf.squeeze(tf.stack(sentences_features, axis=1), axis=-1)
    print(sentences_features_stack)
    best_sentences_scores, _ = tf.math.top_k(sentences_features_stack, k=3)
    
    score = tf.keras.layers.Dense(1)(best_sentences_scores)
    
    return tf.keras.models.Model(inputs=[input_query, input_doc, input_query_idf], outputs=score)


def deep_rank_extra_features(max_q_length,
                              max_s_length,
                              max_s_per_q_term,
                              emb_matrix,
                              filters=16,
                              gru=16,
                              q_term_weight_mode=0,
                              aggregation_mode=0,
                              extraction_mode=0):
    """
    q_term_weight_mode: 0 - use term aggregation with embeddings
                        1 - use term aggregation with idf
                        
    aggregation_mode: 0 - use Bidirectional GRU
                      1 - use Bidirectional GRU + sig for sentence score follow another Bidirectional GRU for aggregation
                      2 - use Bidirectional GRU + sig for sentence score
                      3 - use GRU + sig for sentence score
    
    extraction_mode: 0 - use CNN + GlobalMaxPool
                     1 - use CNN + [GlobalMaxPool, GlobalAvgPool]
                     2 - use CNN + [GlobalMaxPool, GlobalAvgPool, GlobalK-maxAvgPool]
                     3 - use CNN + [GlobalMaxPool, GlobalK-maxAvgPool]

    """

    return_embeddings = q_term_weight_mode==0
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32") # (None, Q)
    input_doc = tf.keras.layers.Input((max_q_length, max_s_per_q_term, max_s_length), dtype="int32") # (None, P, S)
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    input_extra_features = tf.keras.layers.Input((4,), dtype="float32") # (None, 4)
    
    interactions = SemanticInteractions(emb_matrix, return_embeddings=return_embeddings)
    
    if extraction_mode==0:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), padding="SAME", activation="selu")
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
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3),padding="SAME", activation="selu")
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
            elif extraction_mode in [2, 3]:
                kmax_x = kmax_avg_pool(x)
                _concat.append(kmax_x)
            x = concatenate(_concat)
            
            return x, query_embeddings
        
    if aggregation_mode==0:
        aggregation_senteces = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru))
        
    elif aggregation_mode==1:

        l1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1, return_sequences=True), merge_mode="sum")
        l2 = tf.keras.layers.Activation('sigmoid')
        l3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru), merge_mode="sum")

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
        
        l1_a = tf.keras.layers.GRU(1, return_sequences=True, activation="sigmoid")

        l3_a = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        
        def aggregation_senteces(x):
            x = l1_a(x)

            x = l3_a(x)
            return x    
    
    else:
        raise RuntimeError("invalid aggregation_mode")
        
    aggregation = TermAggregation()
    
    def output_score(x):
        tf.keras.layers.Dense(8, activation="selu")(x)
        return tf.keras.layers.Dense(1, activation="relu")(x)
    
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
    doc_vector = aggregation([output_i_stack, query_embeddings])
    
    # score
    score = output_score(doc_vector)
    
    # concat extra features
    score_more_features = tf.keras.layers.Concatenate(axis=-1)([score, input_extra_features])
    
    score_more_features = tf.keras.layers.Dense(8, activation="selu")(score_more_features)
    score_more_features = tf.keras.layers.Dense(1, activation="selu")(score_more_features)
    
    return tf.keras.models.Model(inputs=[input_query, input_doc, input_query_idf, input_extra_features], outputs=score_more_features)
        
        
   