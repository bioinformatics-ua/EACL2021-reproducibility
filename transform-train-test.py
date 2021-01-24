import argparse
import os
import tensorflow as tf
import numpy as np

from timeit import default_timer as timer

from mmnrm.dataset import TrainCollectionV2, sentence_splitter_builderV2
from mmnrm.utils import set_random_seed, load_model_config, load_sentence_generator
from mmnrm.callbacks import Callback
from mmnrm.training import PairwiseTraining
    
# transformer imports
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
from transformers import LongformerTokenizer

from longformerClassifier import TFLongformerForSequenceClassification

LIST_OF_VALID_MODELS = ['distilbert', 'bert_x12', 'bert_x24', 'albert_v2_x12', 'longformer_x12', 'longformer_x24']

def load_transformer(model_type):
    if model_type=="distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
    elif model_type=="bert_x12":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
    elif model_type=="bert_x24":
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = TFBertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=1)
    elif model_type=="albert_v2_x12":
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = TFAlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=1)
    elif model_type=="longformer_x12":
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=1)
    elif model_type=="longformer_x24":
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-large-4096')
        model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-large-4096", num_labels=1)
    else:
        raise ValueError(model_type+" was invalid")
    
    return model, tokenizer
    
def load_data_generator(batch_size, max_len):
    # Load our model tokenizer to do the sentence splitting!

    model_cfg = load_model_config("download_folder/models/still-butterfly-1_batch0_map")
    ## load tokenizer
    train_input_generator, _, our_tokenizer = load_sentence_generator(model_cfg, return_tk=True)

    
    def reverse_input_generator(data_generator):
    
        ## apply the same sentence_generator that we apply to our model
        data_generator = train_input_generator(data_generator)

        for pos_doc, neg_doc in data_generator:

            #just use Y
            pos_doc_queries = our_tokenizer.sequences_to_texts(pos_doc[0])
            neg_doc_queries = our_tokenizer.sequences_to_texts(neg_doc[0])

            pos_tokenized_docs = pos_doc[1].reshape(batch_size,-1,30)
            neg_tokenized_docs = neg_doc[1].reshape(batch_size,-1,30)

            input_text_pos = []
            input_text_neg = []

            for i in range(len(pos_tokenized_docs)):
                input_text_pos.append(pos_doc_queries[i]+tokenizer.sep_token+" ".join(list(filter(lambda x: x != "", our_tokenizer.sequences_to_texts(pos_tokenized_docs[i])))))
                input_text_neg.append(neg_doc_queries[i]+tokenizer.sep_token+" ".join(list(filter(lambda x: x != "", our_tokenizer.sequences_to_texts(neg_tokenized_docs[i])))))

            encoded_input_pos = tokenizer.batch_encode_plus(
                  input_text_pos,
                  max_length=max_len,
                  truncation=True,
                  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                  return_token_type_ids=True,
                  padding="max_length",
                  return_attention_mask=True,
                  return_tensors='tf',  # Return tf tensors
            )

            encoded_input_neg = tokenizer.batch_encode_plus(
                  input_text_neg,
                  max_length=max_len,
                  truncation=True,
                  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                  return_token_type_ids=True,
                  padding="max_length",
                  return_attention_mask=True,
                  return_tensors='tf',  # Return tf tensors
            )

            yield encoded_input_pos, encoded_input_neg


    return TrainCollectionV2\
                            .load("download_folder/pickle-data/training_data_for_EACL")\
                            .batch_size(batch_size)\
                            .set_transform_inputs_fn(reverse_input_generator)


class RecordTimes(Callback):
    def __init__(self, **kwargs):
        self.times=[]

    def on_step_end(self, training_obj, epoch, step, loss, time):
        self.times.append(time)
        print("\rEvaluation {} | time {:.3f} | avg {:.3f} +/- {:.3f} | median {:.3f}".format(step, self.times[-1], 
                                                                                               np.mean(self.times),
                                                                                               np.std(self.times),
                                                                                               np.median(self.times),
                                                                                               end="\r"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This is program will evaluate the time elapsed by several transform based models on tensorflow')
    parser.add_argument('model_type', type=str, choices=LIST_OF_VALID_MODELS, help="type of the model that will be instantiated")
    parser.add_argument('batch_size', type=int, help="size of the batch that will be used during the tests")
    parser.add_argument('-o', type=str, default=None, help="output file to append the results")
    parser.add_argument('-classifier', dest='only_train_classifier', action='store_true', help="only train the classifier" )
    
    args = parser.parse_args()
    
    model, tokenizer = load_transformer(args.model_type)
    
    if args.only_train_classifier:
        model.layers[0].trainable=False
    
    model.summary()
    
    print("max length", tokenizer.model_max_length)
    
    data_generator = load_data_generator(args.batch_size, tokenizer.model_max_length )
        
    callback_times = RecordTimes()

    #static_model = untraced_static_model.get_concrete_function(**next(test_collection.generator()))
    #static_model(**next(test_collection.generator()))

    @tf.function
    def clip_grads(grads):
        gradients, _ = tf.clip_by_global_norm(grads, 5.0)
        return gradients


    train = PairwiseTraining(model=model,
                                 train_collection=data_generator,
                                 grads_callback=clip_grads,
                                 optimizer=tf.keras.optimizers.Adam(),
                                 callbacks=[callback_times])

    
    # bc hugginfaces returns tuples as model output
    train.train(1, draw_graph=False, custom_output=lambda x:x[0])   
    
    times = callback_times.times[1:]
    
    if args.o is not None:
        with open(args.o, "a") as f:
            f.write("{},{},{:.3f},{:.3f},{:.3f}\n".format(args.model_type,
                                                     args.batch_size,
                                                     np.mean(times[:-1]),
                                                     np.std(times[:-1]),
                                                     np.median(times[:-1])))
        
    