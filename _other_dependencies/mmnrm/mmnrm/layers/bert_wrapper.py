import bert

import tensorflow as tf
from tensorflow.keras import backend as K

def load_bert_model(name_model, max_seq_len, trainable=False):
    """
    models name supported, same as tf-2.0-bert
    """
    model_name   = name_model
    model_dir    = bert.fetch_tfhub_albert_model(model_name, ".models")
    model_params = bert.albert_params(model_name)

    l_bert = bert.BertModelLayer.from_params(model_params, name=name_model)

    l_input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32')

    output = l_bert(l_input_ids)                     # output: [batch_size, max_seq_len, hidden_size]

    model = tf.keras.Model(inputs=l_input_ids, outputs=output)
    model.build(input_shape=(None, max_seq_len))

    # load google albert original weights after the build
    bert.load_albert_weights(l_bert, model_dir)
    model.trainable = trainable
    
    
    return model
    
    
