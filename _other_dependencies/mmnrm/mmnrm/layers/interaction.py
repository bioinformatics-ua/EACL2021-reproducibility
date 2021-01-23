import tensorflow as tf
from tensorflow.keras import backend as K

from mmnrm.layers.transformations import ResidualContextLSTM

"""
All layers of this module produce an interaction matrix from a query and a sentence 
"""

class MatrixInteraction(tf.keras.layers.Layer):
    
    def __init__(self, dtype="float32", **kwargs):
        super(MatrixInteraction, self).__init__(dtype=dtype, **kwargs)
        
    def build(self, input_shape):
        # capture the query and document sizes because they are fixed
        
        self.query_max_elements = input_shape[0][1]
        self.setence_max_elements = input_shape[1][1]
        
        super(MatrixInteraction, self).build(input_shape)
        
    def _query_sentence_vector_to_matrices(self, query_vector, sentence_vector):
        """
        Auxiliar function that will replicate each vector in order to procude a matrix with size [B,Q,S]
        where B is batch dimension, Q max terms in the query and S max terms in the sentence
        """
        query_matrix = K.expand_dims(query_vector, axis=2)
        query_matrix = K.repeat_elements(query_matrix, self.setence_max_elements, axis=2)
        
        sentence_matrix = K.expand_dims(sentence_vector, axis=1)
        sentence_matrix = K.repeat_elements(sentence_matrix, self.query_max_elements, axis=1)
        
        return query_matrix, sentence_matrix

class MatrixInteractionMasking(MatrixInteraction):
    """
    Class that handle the computation of the interaction matrix mask 
    """
    
    def __init__(self, mask_value=0, use_mask=True, **kwargs):
        super(MatrixInteractionMasking, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.use_mask = use_mask
    
    def compute_mask_embedding(self, embedding):
        return K.not_equal(embedding, self.mask_value)
    
    def compute_mask_dtype(self, x):
        mask = self.compute_mask(x)
        if mask is not None:
            mask = K.cast(mask, dtype=self.dtype)
            
        return mask
    
    def compute_mask(self, x, mask=None):
        if self.use_mask:
            query_mask = K.cast(self.compute_mask_embedding(x[0]), dtype=self.dtype)
            document_mask = K.cast(self.compute_mask_embedding(x[1]), dtype=self.dtype)

            return K.cast(tf.einsum("bq,bd->bqd", query_mask, document_mask), dtype="bool")
        else:
            return None
    
    
class ExactInteractions(MatrixInteractionMasking): 
    
    def __init__(self, **kwargs):
        super(ExactInteractions, self).__init__(**kwargs)
    
    def call(self, x):
        """
        x[0] - padded query tokens id's
        x[1] - padded sentence tokens id's
        if use_term_importace
            x[2] - query vector with term importances (TF-IDF)
            x[3] - sentence vector with term importances (TF-IDF)
        """
        
        assert(len(x) in [2,4]) # sanity check
        
        mask = self.compute_mask_dtype(x)
        
        query_matrix, sentence_matrix = self._query_sentence_vector_to_matrices(x[0], x[1])
        
        interaction_matrix = K.cast(K.equal(query_matrix, sentence_matrix), dtype=self.dtype)
        if mask is not None:
            interaction_matrix = interaction_matrix * mask
            
        interaction_matrix = K.expand_dims(interaction_matrix)
        
        if len(x)==4:
            query_importance_matrix, sentence_importance_matrix = self._query_sentence_vector_to_matrices(x[2], x[3])
            
            query_importance_matrix = query_importance_matrix * mask
            sentence_importance_matrix = sentence_importance_matrix * mask
            
            interaction_matrix = K.concatenate([interaction_matrix,
                                                K.expand_dims(query_importance_matrix),
                                                K.expand_dims(sentence_importance_matrix)])
        
        return interaction_matrix

class SemanticBaseInteraction(MatrixInteractionMasking):
    """Abstract class should not be initialized"""
    def __init__(self, 
                 learn_term_weights=False,
                 initializer='glorot_uniform',
                 regularizer=None,
                 return_embeddings = False,
                 **kwargs):
        
        super(SemanticBaseInteraction, self).__init__(**kwargs)
        
        self.learn_term_weights = learn_term_weights
        self.initializer = initializer
        self.regularizer = regularizer
        self.return_embeddings = return_embeddings
        
    def _forward_interaction_matrix(self, query_embeddings, sentence_embeddings, mask):
        
        
        interaction_matrix = tf.einsum("bqe,bde->bqd", query_embeddings, sentence_embeddings) 
        
        if mask is not None:
            interaction_matrix = interaction_matrix * mask
            
        interaction_matrix = K.expand_dims(interaction_matrix)
        
        if self.learn_term_weights:
            mask = K.expand_dims(mask)
            
            query_projection = K.dot(query_embeddings, self.query_w)
            sentence_projection = K.dot(sentence_embeddings, self.sentence_w)
            
            query_projection_matrix, sentence_projection_matrix = self._query_sentence_vector_to_matrices(query_projection, sentence_projection) 
            
            if mask is not None:
                query_projection_matrix = query_projection_matrix * mask
                sentence_projection_matrix = sentence_projection_matrix * mask
                
            interaction_matrix = K.concatenate([interaction_matrix, query_projection_matrix, sentence_projection_matrix])
            
        return interaction_matrix


class SemanticInteractions(SemanticBaseInteraction):
    
    def __init__(self, embedding_matrix, trainable_embeddings=False, learn_context=False, **kwargs):
        super(SemanticInteractions, self).__init__(**kwargs)
        # embedding layer as sugested in https://github.com/tensorflow/tensorflow/issues/31086
        # instead of use keras.Embedding
        if trainable_embeddings:
            self.embeddings = tf.Variable(embedding_matrix, dtype=self.dtype)
            self._trainable_weights += [self.embeddings]
        else:    
            self.embeddings = tf.constant(embedding_matrix, dtype=self.dtype)

        self.embedding_dim = embedding_matrix.shape[1]
        
        self.learn_context = learn_context
        
        if self.learn_context:
            self.context_encoder_q = ResidualContextLSTM(self.embedding_dim)
            self.context_encoder_s = ResidualContextLSTM(self.embedding_dim)
        
        print("[EMBEDDING MATRIX SHAPE]", embedding_matrix.shape)
    
    def build(self, input_shape):
        
        # add some weight that will learn term importance projection of the query and sentence embeddings
        if self.learn_term_weights:
            self.query_w = self.add_weight(name="query_w",
                                           shape=(self.embedding_dim, 1),
                                           initializer=self.initializer,
                                           regularizer=self.regularizer,
                                           trainable=True)
            
            self.sentence_w = self.add_weight(name="sentence_w",
                                              shape=(self.embedding_dim, 1),
                                              initializer=self.initializer,
                                              regularizer=self.regularizer,
                                              trainable=True)
        
        super(SemanticInteractions, self).build(input_shape) 
    
    def call(self, x):
        
        """
        x[0] - padded query tokens id's
        x[1] - padded sentence tokens id's
        """
        
        
        mask = self.compute_mask_dtype(x)
        
        # embbed the tokens
        query_embeddings = tf.nn.embedding_lookup(self.embeddings, x[0])
        sentence_embeddings = tf.nn.embedding_lookup(self.embeddings, x[1])
        
        if self.learn_context:
            query_embeddings = self.context_encoder_q(query_embeddings, mask=self.compute_mask_embedding(x[0]))
            sentence_embeddings = self.context_encoder_s(sentence_embeddings, mask=self.compute_mask_embedding(x[1]))
        
        # compute interaction matrix and the term weights
        interaction_matrix = self._forward_interaction_matrix(query_embeddings, sentence_embeddings, mask)
        
        if self.return_embeddings:
            return [interaction_matrix, query_embeddings, sentence_embeddings]
        else:
            return interaction_matrix


        
        
class ContextedSemanticInteractions(SemanticBaseInteraction):
    
    def __init__(self, 
                 context_embedding_layer = None,
                 cls_token_id = None,
                 sep_token_id = None,
                 pad_token_id = None,
                 **kwargs):
        
        super(ContextedSemanticInteractions, self).__init__(**kwargs)

        self.context_embedding_layer = context_embedding_layer
        self.learn_term_weights = learn_term_weights
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.initializer = initializer
        self.regularizer = regularizer
        self.return_embeddings = return_embeddings
        
    def build(self, input_shape):
        
        # add some weight that will learn term importance projection of the query and sentence embeddings
        if self.learn_term_weights:
            assert(len(input_shape)==2)  # sanity check
            
            if self.context_embedding_layer is not None:
                embedding_dim = self.context_embedding_layer.output.shape[-1]
            else:
                embedding_dim = input_shape[-1] # TODO validate this may be necessary to convert into integer
            
            self.query_w = self.add_weight(name="query_w",
                                           shape=(embedding_dim, 1),
                                           initializer=self.initializer,
                                           regularizer=self.regularizer,
                                           trainable=True)
            
            self.sentence_w = self.add_weight(name="sentence_w",
                                              shape=(embedding_dim, 1),
                                              initializer=self.initializer,
                                              regularizer=self.regularizer,
                                              trainable=True)
        
        super(ContextedSemanticInteractions, self).build(input_shape)
        
    def _produce_context_embeddings(self, query_vector, sentence_vector):
        # assume transformer layer follows a BERT architecture
        assert(self.cls_token_id is not None and self.sep_token_id is not None and self.pad_token_id is not None)
        
        batch_dim = tf.shape(query_vector)[0]
        
        cls_input = K.expand_dims(tf.ones((batch_dim,), dtype="int32")*tf.constant(self.cls_token_id))
        sep_input = K.expand_dims(tf.ones((batch_dim,), dtype="int32")*tf.constant(self.sep_token_id))
        
        _input = K.concatenate([cls_input, query_vector, sep_input, sentence_vector, sep_input])

        _out = self.context_embedding_layer(_input)
        
        context_query = _out[:,1:self.query_max_elements+1,:]
        context_sentence = _out[:,self.query_max_elements+2:self.query_max_elements+2+self.setence_max_elements,:]
        
        return context_query, context_sentence
        
    def call(self, x, mask=None):
        """
        x[0] - padded query tokens id's
        x[1] - padded sentence tokens id's
        
        or 
        
        For this setting the mask is needed
        x[0] - padded query context embeddings
        x[1] - padded sentence context embeddings
        
        or
        
        For this setting the mask is needed
        x    - pre-computed similarity matrix, dims (B,Q,D)
        """

        if len(x)==1:
            # precomputed matrix
            assert mask is not None
            interaction_matrix = x * mask
        elif len(x)==2:
            mask = self.compute_mask_dtype(x)
            
            # get query and sentence context embeddings
            if self.context_embedding_layer is not None:
                query_context_embeddings, sentence_context_embeddings = self._produce_context_embeddings(x[0], x[1]) 
            else:
                query_context_embeddings, sentence_context_embeddings = (x[0],x[1])
                
            # compute interaction matrix and the term weights
            interaction_matrix = self._forward_interaction_matrix(query_context_embeddings, sentence_context_embeddings, mask)
            
        else:
            raise NotImplementedError("Missing implmentation when input has len", len(x))
        
        if self.return_embeddings:
            return interaction_matrix, query_context_embeddings, sentence_context_embeddings
        else:
            return interaction_matrix
