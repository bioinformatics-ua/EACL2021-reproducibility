import tensorflow as tf
from tensorflow.keras import backend as K

class DocumentFixWindowSplit(tf.keras.layers.Layer):
    
    def __init__(self, window_size = 10, mask_value = 0., **kwargs):
        super(DocumentFixWindowSplit, self).__init__(**kwargs)
        self.window_size = window_size
        self.mask_value = mask_value
        
    def build(self, input_shape):
        self.num_splites = input_shape[1]//self.window_size
        super(DocumentFixWindowSplit, self).build(input_shape) 
        
    def call(self, x):
        return tf.transpose(tf.split(x, self.num_splites, axis=1), perm=[1,0,2])
    
    def compute_mask(self, inputs, mask=None):
        return tf.transpose(tf.split(tf.not_equal(inputs, self.mask_value), self.num_splites, axis=1), perm=[1,0,2])
    
    
class IdentityMask(tf.keras.layers.Layer):
    
    def __init__(self, mask_value = 0., **kwargs):
        super(IdentityMask, self).__init__(**kwargs)
        self.mask_value = mask_value
        
    def call(self, x, mask=None):
        return x
    
    def compute_mask(self, inputs):
        return tf.not_equal(inputs, self.mask_value)
    
    
class MaskedConcatenate(tf.keras.layers.Layer):
    """
    Concatenation of a list of tensor with a custom beahaviour for the mask
    """
    def __init__(self, index_to_keep, **kwargs):
        """
        Corrent behaviour will return the mask of the input that corresponds to the index_to_keep var
        """
        super(MaskedConcatenate, self).__init__(**kwargs)
        self.index_to_keep = index_to_keep
        
    def call(self, x, mask=None):
        return K.concatenate(x)
    
    def compute_mask(self, x, mask=None):
        assert(isinstance(mask, list))
        return mask[self.index_to_keep]

class ResidualContextLSTM(tf.keras.layers.Layer):
    def __init__(self, size, activation="relu", **kwargs):
        super(ResidualContextLSTM, self).__init__(**kwargs)
        self.lstm = tf.keras.layers.LSTM(size, activation=activation, return_sequences=True)
        
    def call(self, x, mask=None):
        context = self.lstm(x, mask=mask)
        return context + x # residual

class ResidualContextBiLSTM(tf.keras.layers.Layer):
    def __init__(self, size, activation="relu", **kwargs):
        super(ResidualContextBiLSTM, self).__init__(**kwargs)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(size, activation=activation, return_sequences=True), merge_mode="ave")
        
    def call(self, x, mask=None):
        context = self.bilstm(x, mask=mask)
        return context + x # residual
    
class ShuffleRows(tf.keras.layers.Layer):
    """
    Shuffle a tensor along the row dimension (Batch, Row, Colums)
    """
        
    def _build_indexes(self, x):
        indices_tensor_shape = K.shape(x)[:-1]
        l = tf.range(indices_tensor_shape[1], dtype="int32")
        l = tf.random.shuffle(l)


        rows_dim =  K.expand_dims(indices_tensor_shape[0])

        l=tf.tile(l, rows_dim)
        return K.expand_dims(tf.reshape(l, indices_tensor_shape))
    
    def call(self, x, mask=None):
        """
        x[0] - tensor matrix
        x[1] - indices that will guide the permutation, this should follow the format:
        
                permutation => (R1,R2,R3,R4)
                indices => [[[R1], [R2], [R3], [R4]]] * batch_size,
                
                where R1 R2 R3 R4 are the permutated index of the rows
                
                example:  [[[1],[2],[3],[0]]]*10

        """
        indices = self._build_indexes(x)

        return tf.gather_nd(x, indices, batch_dims=1)

        
    
    def compute_mask(self, x, mask=None):
        return None
        #return tf.gather_nd(mask, x[1], batch_dims=1)
    
class ReplaceValuesByThreashold(tf.keras.layers.Layer):
    """
    Replace the values of a tensor given a condition, i.e. a boolean tensor with the same shape of the tensor
    """
    def __init__(self, threshold, replace_value=0, return_filter_mask=False, **kwargs):
        super(ReplaceValuesByThreashold, self).__init__(**kwargs)
        self.threshold = threshold
        self.replace_value = replace_value
        self.return_filter_mask = return_filter_mask
        
    def call(self, x):
        
        filter_mask = tf.cast(x<self.threshold, dtype = tf.int32) 
        
        # ensure performance
        if self.replace_value!=0:
            x = x*filter_mask + self.replace_value*(1-filter_mask)
        else:
            x = x*filter_mask
            
        if self.return_filter_mask:
            return x, filter_mask
        else:
            return x
        
class ReplaceValuesByMask(tf.keras.layers.Layer):
    """
    Replace the values of a tensor given a mask
    """
    def __init__(self, replace_value=0, **kwargs):
        super(ReplaceValuesByMask, self).__init__(**kwargs)
        self.replace_value = replace_value

    def call(self, x):
        
        filter_mask = K.cast(x[1], dtype=x[0].dtype)
        x = x[0]
        
        # ensure performance
        if self.replace_value!=0:
            x = x*filter_mask + self.replace_value*(1-filter_mask)
        else:
            x = x*filter_mask

        return x    

class GlobalKmaxAvgPooling2D(tf.keras.layers.Layer):
    """
    Aplies k-max avg pooling to a 4D tensor
    """
    def __init__(self, kmax=5, **kwargs):
        super(GlobalKmaxAvgPooling2D, self).__init__(**kwargs)
        self.kmax = kmax

    def build(self, input_shape):
        self.filter_dim = input_shape[-1]
        super(GlobalKmaxAvgPooling2D, self).build(input_shape) 
    
    def call(self, x):

        x = tf.reshape(x, (K.shape(x)[0],-1,self.filter_dim))
        x = tf.linalg.matrix_transpose(x)
        x_kmax, _ = tf.math.top_k(x, k=self.kmax, sorted=False)

        return K.mean(x_kmax, axis=-1)
    
class GlobalMaskedAvgPooling2D(tf.keras.layers.Layer):
    """
    Aplies avg pooling to a 4D tensor using a mask
    """

    def call(self, x, mask=None):
        assert(mask is not None)
        mask = K.expand_dims(K.cast(mask, dtype=self.dtype))
        x = x * mask
        mask_elements = tf.math.reduce_sum(mask, axis=[1,2])
        sum_x = tf.math.reduce_sum(x, axis=[1,2])
        return sum_x/mask_elements
    
    def compute_mask(self, x, mask=None):
        return None
    
class GlobalKmax2D(tf.keras.layers.Layer):
    """
    Aplies kmax pooling to a 4D tensor
    """
    
    def __init__(self, kmax=3, **kwargs):
        super(GlobalKmax2D, self).__init__(**kwargs)
        self.kmax = kmax

    def build(self, input_shape):
        self.filter_dim = input_shape[-1]
        super(GlobalKmax2D, self).build(input_shape) 
    
    def call(self, x): #B, Q, D, F
        batch_size = K.shape(x)[0]
        
        x = tf.reshape(x, (batch_size,-1,self.filter_dim)) # B, QxD, F
        x = tf.linalg.matrix_transpose(x) # B, F, QxD
        x, _ = tf.math.top_k(x, k=self.kmax, sorted=False) # B, F, MAX

        return tf.reshape(x, (batch_size,self.kmax*self.filter_dim)) # B, F x MAX
    
    def compute_mask(self, x, mask=None):
        return None
 