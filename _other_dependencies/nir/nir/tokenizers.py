
"""
This class is a simplification of the tensorflow keras tokenizer with some changes
But the original source code can be found on: https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/keras/preprocessing/text/Tokenizer
"""

from collections import OrderedDict
from collections import defaultdict
import tempfile
import pickle
import shutil
import json
import os
from multiprocessing import Process
import gc
import sys
import re
from nir.logger import log
from os.path import join
import types

def fitTokenizeJob(proc_id, articles, _class, merge_tokenizer_path, properties):
    print("[Process-{}] Started".format(proc_id))
    sys.stdout.flush()
    # ALL THREADS RUN THIS
    tk = _class(**properties)
    tk.fit_on_texts(articles)
    del articles

    file_name = "tk_{0:03}.p".format(proc_id)
    print("[Process-{}]: Store {}".format(proc_id, file_name))

    tk.save_to_json(path=os.path.join(merge_tokenizer_path, file_name))
    del tk
    print("[Process-{}] Ended".format(proc_id))


def tokenizeJob(proc_id, texts, _class, tokenazer_info, merge_tokenizer_path, properties, kwargs):
    print("[Process-{}] Started articles size {}".format(proc_id, len(texts)))

    sys.stdout.flush()
    # load tokenizer
    tokenizer = _class(cache_folder=tokenazer_info["cache_folder"],
                       prefix_name=tokenazer_info["prefix_name"])

    tokenizer.word_index = tokenazer_info["word_index"]
    tokenizer.oov_token = tokenazer_info["oov_token"]
    tokenizer.num_words = tokenazer_info["num_words"]
    
    # ALL THREADS RUN THIS
    tokenized_texts = tokenizer.tokenize_texts(texts, **kwargs)
    del texts

    file_name = "tokenized_text_{0:03}.p".format(proc_id)
    print("[Process-{}]: Store {}".format(proc_id, file_name))

    with open(os.path.join(merge_tokenizer_path, file_name), "wb") as f:
        pickle.dump(tokenized_texts, f)

    del tokenized_texts
    gc.collect()

    print("[Process-{}] Ended".format(proc_id))


class BaseTokenizer:
    """Text tokenization utility class.

    This class allows to vectorize a text corpus, by turning each
    text into either a sequence of integers (each integer being the index
    of a token in a dictionary)

    # Arguments
        num_words: the maximum number of words to keep, based
            on word frequency. Only the most common `num_words-1` words will
            be kept.
        filters: a string where each element is a character that will be
            filtered from the texts. The default is all punctuation, plus
            tabs and line breaks, minus the `'` character.
        lower: boolean. Whether to convert the texts to lowercase.
        split: str. Separator for word splitting.
        char_level: if True, every character will be treated as a token.
        oov_token: if given, it will be added to word_index and used to
            replace out-of-vocabulary words during text_to_sequence calls

    `0` is a reserved index that won't be assigned to any word.
    """

    def __init__(self, cache_folder,
                 prefix_name,
                 num_words=None,
                 min_word_frequency=None,
                 oov_token=None,
                 document_count=0,
                 n_process=4,
                 **kwargs):

        self.prefix_name = prefix_name
        self.cache_folder = cache_folder
        if 'word_counts' in kwargs:
            self.word_counts = json.loads(kwargs.pop('word_counts'))
        else:
            self.word_counts = OrderedDict()
        if 'word_docs' in kwargs:
            self.word_docs = json.loads(kwargs.pop('word_docs'))
            self.index_docs = {int(d): w for w, d in self.word_docs.items()}
        else:
            self.word_docs = {}
            self.index_docs = {}
            
        self.num_words = num_words
        self.min_word_frequency = min_word_frequency

        self.document_count = document_count
        self.oov_token = oov_token

        if 'word_index' in kwargs:
            self.word_index = json.loads(kwargs.pop('word_index'))
            self.index_word = {int(i): w for w, i in self.word_index.items()}
        else:
            self.word_index = {}
            self.index_word = {}
        
        self.n_process = n_process
        
        if kwargs:
            print('[WARNING] Unrecognized keyword arguments: ' + str(kwargs.keys()))

    def tokenizer(self):
        raise NotImplementedError("The function tokenizer must be implemented by a subclass")

    def save_to_json(self, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def load_from_json(path):
        raise NotImplementedError()
    
    def get_vocabulary(self):
        if self.num_words is not None:
            return [k for k,v in self.word_index.items() if v<(self.num_words)]
        else:
            return list(self.word_index.keys())
   
    def vocabulary_size(self):
        return len(self.get_vocabulary())
    
    
    def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.

        In the case where texts contains lists,
        we assume each entry of the lists to be a token.

        Required before using `texts_to_sequences` or `texts_to_matrix`.

        # Arguments
            texts: can be a list of strings,
                a generator of strings (for memory-efficiency),
                or a list of list of strings.
        """
        for text in texts:
            self.document_count += 1
            if isinstance(text, list):
                raise ValueError("Found list instead of a string")
            else:
                seq = self.tokenizer(text)
                
                for w in seq:
                    if w in self.word_counts:
                        self.word_counts[w] += 1
                    else:
                        self.word_counts[w] = 1
                for w in set(seq):
                    # In how many documents each word occurs
                    if w in self.word_docs:
                        self.word_docs[w] += 1
                    else:
                        self.word_docs[w] = 1

        self._build_tokenizer()

    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words

        for text in texts:
            if isinstance(text, list):
                raise ValueError("Found list instead of a string")
            else:
                seq = self.tokenizer(text)

            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        continue
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(self.word_index.get(self.oov_token))
            yield vect

    def sequences_to_texts(self, sequences):
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None:
                    if num_words and num >= num_words:
                        if oov_token_index is not None:
                            vect.append(self.index_word[oov_token_index])
                    else:
                        vect.append(word)
                elif self.oov_token is not None:
                    vect.append(self.index_word[oov_token_index])
            vect = ' '.join(vect)
            yield vect

    def is_trained(self):
        return len(self.word_counts) > 0

    def get_config(self):
        '''Returns the tokenizer configuration as Python dictionary.
        The word count dictionaries used by the tokenizer get serialized
        into plain JSON, so that the configuration can be read by other
        projects.

        # Returns
            A Python dictionary with the tokenizer configuration.
        '''
        json_word_counts = json.dumps(self.word_counts)
        json_word_docs = json.dumps(self.word_docs)
        json_word_index = json.dumps(self.word_index)

        return {
            'cache_folder': self.cache_folder,
            'prefix_name': self.prefix_name,
            'num_words': self.num_words,
            'min_word_frequency': self.min_word_frequency,
            'oov_token': self.oov_token,
            'document_count': self.document_count,
            'n_process' : self.n_process,
            'word_counts': json_word_counts,
            'word_docs': json_word_docs,
            'word_index': json_word_index
        }
    
    def update_min_word_frequency(self, min_word_frequency):
        
        self.min_word_frequency = min_word_frequency
        self.num_words = len(self.index_word)+1
        
        if self.min_word_frequency is None:
            self.num_words = None
        else:
            
            for i in range(1, len(self.index_word)+1):
                if self.min_word_frequency > self.word_counts[self.index_word[i]]:
                    self.num_words = i
                    break
                    
    
    def _build_tokenizer(self):
        # CODE FROM KERAS TOKENIZER
        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
        

            
        # calculate the num_words based on min_word_frequency
        if self.min_word_frequency is not None:
            for i,w in enumerate(sorted_voc):
                if self.min_word_frequency > self.word_counts[w]:
                    self.num_words = i
                    break

        self.index_word = dict((c, w) for w, c in self.word_index.items())

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c
    
    def fit_tokenizer_multiprocess(self, corpora_iterator):
        merge_tokenizer_path = tempfile.mkdtemp()

        try:
            # initialization of the process
            def fitTokenizer_process_init(proc_id, articles):
                return Process(target=fitTokenizeJob, args=(proc_id, articles, self.__class__, merge_tokenizer_path, self.get_config()))

            # multiprocess loop
            for i, texts in enumerate(corpora_iterator):
                process = []

                t_len = len(texts)
                t_itter = t_len//self.n_process

                for k, j in enumerate(range(0, t_len, t_itter)):
                    process.append(fitTokenizer_process_init(self.n_process*i+k, texts[j:j+t_itter]))

                print("[MULTIPROCESS LOOP] Starting", self.n_process, "process")
                for p in process:
                    p.start()

                print("[MULTIPROCESS LOOP] Wait", self.n_process, "process")
                for p in process:
                    p.join()
                gc.collect()

            # merge the tokenizer
            print("[TOKENIZER] Merge")
            files = sorted(os.listdir(merge_tokenizer_path))

            for file in files:
                log.info("[TOKENIZER] Load {}".format(file))
                loaded_tk = self.__class__.load_from_json(path=os.path.join(merge_tokenizer_path, file), **self.get_config())

                # manual merge
                for w, c in loaded_tk.word_counts.items():
                    if w in self.word_counts:
                        self.word_counts[w] += c
                    else:
                        self.word_counts[w] = c

                for w, c in loaded_tk.word_docs.items():
                    if w in self.word_docs:
                        self.word_docs[w] += c
                    else:
                        self.word_docs[w] = c

                self.document_count += loaded_tk.document_count

            self._build_tokenizer()

            # Saving tokenizer
            self.save_to_json()
        except Exception as e:
            raise e
        finally:
            # always remove the temp directory
            log.info("[TOKENIZER] Remove {}".format(merge_tokenizer_path))
            shutil.rmtree(merge_tokenizer_path)

    def tokenizer_multiprocess(self, gen_texts, n_process=None, **kwargs):
        
        
        if isinstance(gen_texts, types.GeneratorType):
            articles = []
            for texts in gen_texts:
                articles.extend(self._tokenizer_multiprocess(texts, n_process=n_process))
            return articles
        else:
            return self._tokenizer_multiprocess(gen_texts, n_process=n_process)
        
            
    def _tokenizer_multiprocess(self, texts, n_process=None, **kwargs):
        
        if n_process is None:
            n_process = self.n_process
        
        merge_tokenizer_path = tempfile.mkdtemp()
        tokenized_texts = []
        
        w_index = {w:i for w,i in self.word_index.items() if i<=self.num_words}
        
        tokenazer_info = {
            "num_words": self.num_words,
            "oov_token": self.oov_token,
            "word_index": w_index,
            'cache_folder': self.cache_folder,
            'prefix_name': self.prefix_name,
        }

        try:
            # initialization of the process
            def tokenizer_process_init(proc_id, texts):
                return Process(target=tokenizeJob, args=(proc_id, texts, self.__class__, tokenazer_info, merge_tokenizer_path, self.get_config(), kwargs))
            
            # multiprocess loop
            itter = 1000000
            for i, l in enumerate(range(0, len(texts), itter)):
                process = []

                docs = texts[l:l+itter]
                t_len = len(docs)
                t_itter = t_len//n_process

                for k, j in enumerate(range(0, t_len, t_itter)):
                    process.append(tokenizer_process_init(i*n_process+k, docs[j:j+t_itter]))

                print("[MULTIPROCESS LOOP] Starting", n_process, "process")
                for p in process:
                    p.start()

                print("[MULTIPROCESS LOOP] Wait", n_process, "process")
                for p in process:
                    p.join()

                del docs
                gc.collect()

            # merge the tokenizer
            print("[TOKENIZER] Merge tokenized files")
            files = sorted(os.listdir(merge_tokenizer_path))
            del texts

            for file in files:
                log.info("[TOKENIZER] Load {}".format(file))
                with open(os.path.join(merge_tokenizer_path, file), "rb") as f:
                    tokenized_texts.extend(pickle.load(f))

        except Exception as e:
            raise e
        finally:
            # always remove the temp directory
            log.info("[TOKENIZER] Remove {}".format(merge_tokenizer_path))
            shutil.rmtree(merge_tokenizer_path)

        return tokenized_texts

    
    
class Regex(BaseTokenizer):
    
    def __init__(self, sw_file=None, queries_sw=False, articles_sw=False, **kwargs):
        super().__init__(**kwargs)

        self.sw_file = sw_file

        if isinstance(queries_sw, str):
            self.queries_sw = queries_sw == "true"
        else:
            self.queries_sw = queries_sw

        if isinstance(articles_sw, str):
            self.articles_sw = articles_sw == "true"
        else:
            self.articles_sw = articles_sw

        if sys.version_info < (3,):
            self.maketrans = string.maketrans
        else:
            self.maketrans = str.maketrans
        
        self.pattern = re.compile('[^a-zA-Z0-9\s]+')
        self.filter_whitespace = lambda x: not x == ""
        
        self.init_name()
        
        print("DEBUG created tokenizer", self.name)
        if self.sw_file is not None:
            with open(self.sw_file, "r") as f:
                self.stop_words = json.load(f)
        self.stop_words_tokenized = None

        print(self.queries_sw, self.articles_sw)
    
    
    def init_name(self):
        self.name = Regex.build_name(self.prefix_name)
        self.name_properties = self.name+str(self.queries_sw)+"_"+str(self.articles_sw)
        
    @staticmethod
    def build_name(prefix_name):
        return prefix_name + "_RegexTokenizer"

    def tokenize_texts(self, texts, **kwargs):
        if "mode" in kwargs:
            if kwargs["mode"] == "queries":
                flag = self.queries_sw
            elif kwargs["mode"] == "articles":
                flag = self.articles_sw
        else:
            flag = False

        tokenized_texts = self.texts_to_sequences(texts)
        if flag:
            if self.stop_words_tokenized is None:  # lazzy initialization
                self.stop_words_tokenized = set(self.texts_to_sequences([self.stop_words])[0])

            for tokenized_text in tokenized_texts:
                tokenized_text = [token for token in tokenized_text if token not in self.stop_words_tokenized]

        return tokenized_texts

    def tokenize_query(self, query):
        tokenized_query = self.texts_to_sequences([query])[0]
        if self.queries_sw:
            if self.stop_words_tokenized is None:  # lazzy initialization
                self.stop_words_tokenized = set(self.texts_to_sequences([self.stop_words])[0])
            tokenized_query = [token for token in tokenized_query if token not in self.stop_words_tokenized]

        return tokenized_query

    def tokenize_article(self, article):
        tokenized_article = self.texts_to_sequences([article])[0]
        if self.articles_sw:
            if self.stop_words_tokenized is None:  # lazzy initialization
                self.stop_words_tokenized = set(self.texts_to_sequences([self.stop_words])[0])
            tokenized_article = [token for token in tokenized_article if token not in self.stop_words_tokenized]

        return tokenized_article

    def tokenizer(self, text):
        
        text = text.lower()

        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        tab = self.maketrans(filters, " "*(len(filters)))
        text = text.translate(tab)
        tokens = self.pattern.sub(' ', text).split(" ")
        tokens = list(filter(self.filter_whitespace, tokens))

        return tokens

    def get_config(self):
        t_config = super().get_config()

        if self.sw_file is not None:
            t_config["sw_file"] = self.sw_file
            t_config["queries_sw"] = self.queries_sw
            t_config["articles_sw"] = self.articles_sw
            
        return t_config

    def save_to_json(self, **kwargs):
        """
        KERAS code

        Returns a JSON string containing the tokenizer configuration.
        To load a tokenizer from a JSON string, use
        `keras.preprocessing.text.tokenizer_from_json(json_string)`.

        # Returns
            A JSON string containing the tokenizer configuration.
        """
        if 'path' in kwargs:
            path = kwargs.pop('path')
        else:
            path = join(self.cache_folder, self.name+".json")
        with open(path, "w") as f:
            json.dump(self.get_config(), f)

    @staticmethod
    def load_from_json(**kwargs):
        if "path" in kwargs:
            path = kwargs["path"]
        elif "cache_folder" in kwargs and "prefix_name" in kwargs:
            path = join(kwargs["cache_folder"], Regex.build_name(kwargs["prefix_name"])+".json")
        else:
            raise ValueError("Path not specified")
            
        with open(path, "r") as f:
            t_config = json.load(f)
            # override - TODO change this is stupid this way
            if "sw_file" in kwargs:
                t_config["queries_sw"] = kwargs["queries_sw"]
                t_config["articles_sw"] = kwargs["articles_sw"]
                t_config["sw_file"] = kwargs["sw_file"]
            return Regex(**t_config)   
        
class BioCleanTokenizer(BaseTokenizer):
    
    def __init__(self, sw_file=None, queries_sw=False, articles_sw=False, **kwargs):
        super().__init__(**kwargs)

        self.sw_file = sw_file

        if isinstance(queries_sw, str):
            self.queries_sw = queries_sw == "true"
        else:
            self.queries_sw = queries_sw

        if isinstance(articles_sw, str):
            self.articles_sw = articles_sw == "true"
        else:
            self.articles_sw = articles_sw

        self.pattern = re.compile('[.,?;*!%^&_+():-\[\]{}]') 

        self.init_name()
        
        print("DEBUG created tokenizer", self.name)
        if self.sw_file is not None:
            with open(self.sw_file, "r") as f:
                self.stop_words = json.load(f)
        self.stop_words_tokenized = None

        print(self.queries_sw, self.articles_sw)
    
    
    def init_name(self):
        self.name = BioCleanTokenizer.build_name(self.prefix_name)
        self.name_properties = self.name+str(self.queries_sw)+"_"+str(self.articles_sw)
        
    @staticmethod
    def build_name(prefix_name):
        return prefix_name + "_BioCleanTokenizer"

    def tokenize_texts(self, texts, **kwargs):
        if "mode" in kwargs:
            if kwargs["mode"] == "queries":
                flag = self.queries_sw
            elif kwargs["mode"] == "articles":
                flag = self.articles_sw
        else:
            flag = False

        tokenized_texts = self.texts_to_sequences(texts)
        if flag:
            if self.stop_words_tokenized is None:  # lazzy initialization
                self.stop_words_tokenized = set(self.texts_to_sequences([self.stop_words])[0])

            for tokenized_text in tokenized_texts:
                tokenized_text = [token for token in tokenized_text if token not in self.stop_words_tokenized]

        return tokenized_texts

    def tokenize_query(self, query):
        tokenized_query = self.texts_to_sequences([query])[0]
        if self.queries_sw:
            if self.stop_words_tokenized is None:  # lazzy initialization
                self.stop_words_tokenized = set(self.texts_to_sequences([self.stop_words])[0])
            tokenized_query = [token for token in tokenized_query if token not in self.stop_words_tokenized]

        return tokenized_query

    def tokenize_article(self, article):
        tokenized_article = self.texts_to_sequences([article])[0]
        if self.articles_sw:
            if self.stop_words_tokenized is None:  # lazzy initialization
                self.stop_words_tokenized = set(self.texts_to_sequences([self.stop_words])[0])
            tokenized_article = [token for token in tokenized_article if token not in self.stop_words_tokenized]

        return tokenized_article

    def tokenizer(self, text):
        return self.pattern.sub('',text.replace('"', '').replace('/', '').replace('\\', '').replace("'",'').strip().lower()).split()

    def get_config(self):
        t_config = super().get_config()

        if self.sw_file is not None:
            t_config["sw_file"] = self.sw_file
            t_config["queries_sw"] = self.queries_sw
            t_config["articles_sw"] = self.articles_sw
            
        return t_config

    def save_to_json(self, **kwargs):
        """
        KERAS code

        Returns a JSON string containing the tokenizer configuration.
        To load a tokenizer from a JSON string, use
        `keras.preprocessing.text.tokenizer_from_json(json_string)`.

        # Returns
            A JSON string containing the tokenizer configuration.
        """
        if 'path' in kwargs:
            path = kwargs.pop('path')
        else:
            path = join(self.cache_folder, self.name+".json")
        with open(path, "w") as f:
            json.dump(self.get_config(), f)

    @staticmethod
    def load_from_json(**kwargs):
        if "path" in kwargs:
            path = kwargs["path"]
        elif "cache_folder" in kwargs and "prefix_name" in kwargs:
            path = join(kwargs["cache_folder"], BioCleanTokenizer.build_name(kwargs["prefix_name"])+".json")
        else:
            raise ValueError("Path not specified")
            
        with open(path, "r") as f:
            t_config = json.load(f)
            # override - TODO change this is stupid this way
            if "sw_file" in kwargs:
                t_config["queries_sw"] = kwargs["queries_sw"]
                t_config["articles_sw"] = kwargs["articles_sw"]
                t_config["sw_file"] = kwargs["sw_file"]
            return BioCleanTokenizer(**t_config)


class BioCleanTokenizer2(BioCleanTokenizer):
    
    def __init__(self, sw_file=None, queries_sw=False, articles_sw=False, **kwargs):
        super().__init__(sw_file, queries_sw, articles_sw, **kwargs)
        self.pattern = re.compile('[.,?;*!%^&_+():\-\[\]{}]')

        

    def init_name(self):
        self.name = BioCleanTokenizer2.build_name(self.prefix_name)
        self.name_properties = self.name+str(self.queries_sw)+"_"+str(self.articles_sw)
    
    @staticmethod
    def build_name(prefix_name):
        return prefix_name + "_BioCleanTokenizer2"
    
    def tokenizer(self, text):
        return self.pattern.sub(' ',text.replace('"', '').replace('/', '').replace('\\', '').replace("'",'').strip().lower()).split()
    
    @staticmethod
    def load_from_json(**kwargs):
        if "path" in kwargs:
            path = kwargs["path"]
        elif "cache_folder" in kwargs and "prefix_name" in kwargs:
            path = join(kwargs["cache_folder"], BioCleanTokenizer2.build_name(kwargs["prefix_name"])+".json")
        else:
            raise ValueError("Path not specified")
            
        with open(path, "r") as f:
            t_config = json.load(f)
            # override - TODO change this is stupid this way
            if "sw_file" in kwargs:
                t_config["queries_sw"] = kwargs["queries_sw"]
                t_config["articles_sw"] = kwargs["articles_sw"]
                t_config["sw_file"] = kwargs["sw_file"]
            return BioCleanTokenizer2(**t_config)
