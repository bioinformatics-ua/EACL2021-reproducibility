import argparse
import os
import tensorflow as tf
import numpy as np

from timeit import default_timer as timer

from mmnrm.dataset import TestCollectionV2, sentence_splitter_builderV2
from mmnrm.utils import set_random_seed, load_model_config, load_sentence_generator

# transformer imports
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
from transformers import LongformerTokenizer

from longformerClassifier import TFLongformerForSequenceClassification

LIST_OF_VALID_MODELS = ['distilbert', 'bert_x12', 'bert_x24', 'albert_v2_x12', 'longformer_x12', 'longformer_512']

def load_transformer(model_type):
    if model_type=="distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
        max_length = tokenizer.model_max_length
    elif model_type=="bert_x12":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
        max_length = tokenizer.model_max_length
    elif model_type=="bert_x24":
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = TFBertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=1)
        max_length = tokenizer.model_max_length
    elif model_type=="albert_v2_x12":
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = TFAlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=1)
        max_length = tokenizer.model_max_length
    elif model_type=="longformer_x12":
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=1)
        max_length = tokenizer.model_max_length
    elif model_type=="longformer_512":
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-large-4096')
        model = TFLongformerForSequenceClassification.from_pretrained("allenai/longformer-large-4096", num_labels=1)
        max_length = 512
    else:
        raise ValueError(model_type+" was invalid")
    
    return model, max_length, tokenizer
    
def load_data_generator(batch_size, max_length, tokenizer):
    # Load our model tokenizer to do the sentence splitting!
    model_cfg = load_model_config("download_folder/models/still-butterfly-1_batch0_map")
    ## load tokenizer
    _, test_input_generator, our_tokenizer = load_sentence_generator(model_cfg, return_tk=True)
    
    def reverse_input_generator(data_generator):

        ## apply the same sentence_generator that we apply to our model
        data_generator = test_input_generator(data_generator)

        for ids, Y, ids_docs, offset_docs in data_generator:

            #just use Y
            queries = our_tokenizer.sequences_to_texts(Y[0])
            tokenized_docs = Y[1].reshape(batch_size,-1,30)

            input_text = []

            for i,tokenized_doc in enumerate(tokenized_docs):
                input_text.append(queries[i]+tokenizer.sep_token+" ".join(list(filter(lambda x: x != "", our_tokenizer.sequences_to_texts(tokenized_doc)))))


            encoded_input = tokenizer.batch_encode_plus(
                  input_text,
                  max_length=max_length,
                  truncation=True,
                  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                  return_token_type_ids=True,
                  padding="max_length",
                  return_attention_mask=True,
                  return_tensors='tf',  # Return tf tensors
            )

            yield encoded_input


    return TestCollectionV2.load("download_folder/pickle-data/query_docs_pairs")\
                           .batch_size(batch_size)\
                           .set_transform_inputs_fn(reverse_input_generator)
    



    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This is program will evaluate the time elapsed by several transform based models on tensorflow')
    parser.add_argument('model_type', type=str, choices=LIST_OF_VALID_MODELS, help="type of the model that will be instantiated")
    parser.add_argument('batch_size', type=int, help="size of the batch that will be used during the tests")
    parser.add_argument('-o', type=str, default=None, help="output file to append the results")

    
    args = parser.parse_args()
    
    model, max_length, tokenizer = load_transformer(args.model_type)
    
    print("max length", max_length)
    
    data_generator = load_data_generator(args.batch_size, max_length, tokenizer )
    

    # use tf.function to build a static graph
    @tf.function
    def untraced_static_model(**kwargs):
        return model(kwargs)

    traced_model = untraced_static_model.get_concrete_function(**next(data_generator.generator()))

    print("A static graph was created")

        
    ## dummy run
    traced_model(**next(data_generator.generator())) 
    
    times = []

    for i,Y in enumerate(data_generator.generator()):

        s_time = timer()

        output = traced_model(**Y)
        dummy = output[0][0].numpy()
        #print(output[0][0].numpy)
        e_time = timer()-s_time
        times.append(e_time)
        
        if not i%2:
            print("\rEvaluation {} | time {:.3f} | avg {:.3f} +/- {:.3f} | median {:.3f}".format(i, 
                                                                                               times[-1], 
                                                                                               np.mean(times),
                                                                                               np.std(times),
                                                                                               np.median(times)),end="\r")
    
    if args.o is not None:
        with open(args.o, "a") as f:
            f.write("{},{},{:.3f},{:.3f},{:.3f}\n".format(args.model_type,
                                                     args.batch_size,
                                                     np.mean(times[:-1]),
                                                     np.std(times[:-1]),
                                                     np.median(times[:-1])))
        
    