"""
This file contains layers that handle and process interactions matrices in order to produce local relevance.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


class MultipleNgramConvs(tf.keras.layers.Layer): 
    
    def __init__(self,
                 max_ngram,
                 k_max,
                 k_polling_avg = 5, # do k_polling avg after convolution
                 polling_avg = True, # do avg polling after convolution
                 use_mask = True,
                 filters=lambda x,y:x*y*2**y, # can be a list or a function of input features and n-gram
                 activation="relu",
                 dtype="float32", 
                 **kwargs):
        super(MultipleNgramConvs, self).__init__(dtype=dtype, **kwargs)
        self.k_max = k_max
        self.max_ngram = max_ngram
        self.k_polling_avg = k_polling_avg
        self.polling_avg = polling_avg
        self.use_mask = use_mask
        self.filters = filters
        self.activation = activation
            
    def build(self, input_shape):
        
        self.input_feature_dim = input_shape[-1]
        
        self.convolutions = []
        
        for n in range(1, self.max_ngram+1):
            
            if callable(self.filters):
                filters = self.filters(self.input_feature_dim, n)
            elif isinstance(self.filters, list):
                filters = self.filters[n]
            elif isinstance(self.filters, int):
                filters = self.filters
            else:
                raise ValueError("filters_function must be a function(x, y) or a list or a integer")
            
            _conv_layer = tf.keras.layers.Conv2D(filters, 
                                                 (n,n),
                                                 activation=self.activation,
                                                 padding="SAME",
                                                 dtype=self.dtype)
            _conv_layer.build(input_shape)
            self._trainable_weights += _conv_layer.trainable_weights # mannualy add the trainable weights
            self.convolutions.append(_conv_layer) 

        super(MultipleNgramConvs, self).build(input_shape)
        
    def call(self, x, mask=None):
        """
        x - should be the output of an interaction matrix, i.e, a 3D tensor (4D if batch is accounted)
        """
        # forward computation
        
        # tensor convolution for the differents ngram (1 to max_ngram).
        multiple_convs = [ conv(x) for conv in self.convolutions ]
        
        polling = []
        for conv in multiple_convs:
            if self.use_mask:
                # Note that the multiplication by the mask here my be redudant due to the a max operation that will follow.
                conv = conv * K.expand_dims(K.cast(mask, dtype=self.dtype))
            
            polling.append(tf.nn.top_k(K.max(conv, axis=-1), k=self.k_max)[0])
            
            if self.polling_avg:
                polling.append(tf.nn.top_k(K.mean(conv, axis=-1), k=self.k_max)[0])
            
            if self.k_polling_avg is not None:
                raise NotImplementedError("The k_polling_avg is currently not working, awaiting for review")
                polling.append(tf.nn.top_k(K.mean(tf.nn.top_k(conv, k=self.k_polling_avg)[0], axis=-1), k=self.k_max)[0])
        
        # _old multiple_conv = [ tf.nn.top_k(K.max(conv(x)*mask, axis=-1), k=self.k_max)[0] for conv in self.convolutions ] 

        concatenate_convs = K.concatenate(polling, axis=-1)
        
        return concatenate_convs
    
    def compute_mask(self, x, mask=None):
        # extract just the query mask
        if self.use_mask:
            return mask[:,:,0]
        else:
            return None

class SimpleMultipleNgramConvs(tf.keras.layers.Layer): 
    
    def __init__(self,
                 max_ngram,
                 filters,
                 activation="relu",
                 dtype="float32", 
                 **kwargs):
        super(SimpleMultipleNgramConvs, self).__init__(dtype=dtype, **kwargs)

        self.max_ngram = max_ngram
        self.filters = filters
        self.activation = activation
            
    def build(self, input_shape):
        
        self.input_feature_dim = input_shape[-1]
        
        self.convolutions = []
        
        for n in range(1, self.max_ngram+1):
            
            _conv_layer = tf.keras.layers.Conv2D(self.filters, 
                                                 (n,n),
                                                 activation=self.activation,
                                                 padding="SAME",
                                                 dtype=self.dtype)
            _conv_layer.build(input_shape)
            self._trainable_weights += _conv_layer.trainable_weights # mannualy add the trainable weights
            self.convolutions.append(_conv_layer) 

        super(SimpleMultipleNgramConvs, self).build(input_shape)
        
    def call(self, x, mask=None):
        """
        x - should be the output of an interaction matrix, i.e, a 3D tensor (4D if batch is accounted)
        """
        # forward computation
        
        # tensor convolution for the differents ngram (1 to max_ngram).
        multiple_convs = [ conv(x) for conv in self.convolutions ]
        
        resulting_covs = []
        for conv in multiple_convs:
            conv = conv * K.expand_dims(K.cast(mask, dtype=self.dtype))
            resulting_covs.append(conv)
            
        return resulting_covs
    
    def compute_mask(self, x, mask=None):
        return None

class MaskedSoftmax(tf.keras.layers.Layer): 
    def __init__(self, mask_value=0, dtype="float32", **kwargs):
        super(MaskedSoftmax, self).__init__(dtype=dtype, **kwargs)
        self.mask_value = mask_value
        
    def call(self, x):
        tf.debugging.assert_type(x, self.dtype)
        
        mask = self.compute_mask(x)
        mask = tf.math.logical_not(mask)

        x -= 1.e9 * K.cast(mask, dtype=self.dtype)

        return K.softmax(x)
    
    def compute_mask(self, x, mask=None):
        return K.not_equal(x, self.mask_value)

    
    