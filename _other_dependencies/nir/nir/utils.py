import sys
import matplotlib.pyplot as plt

def create_filter_query_function():
    if sys.version_info < (3,):
        maketrans = string.maketrans
    else:
        maketrans = str.maketrans
    filters = '+-=&|><!(){}[]^"~*?:\/'
    tab = maketrans(filters, " "*(len(filters)))
    
    def f(query_string):
        return query_string.translate(tab)
    return f


def try_function_n_times(f, num_tries):
    last_excepetion = None
    for _ in range(num_tries):
        try:
            f()
            break
        except Exception as e:
            last_excepetion = e
            # fail but lets try again
            print("function",f.__name__,"fail",_,"times")
            
    if _==num_tries-1:
        if last_excepetion is not None:
            raise last_excepetion # excedeed number of tries, so raise the exception
        else:
            raise RuntimeError("Number of calls were exceeded")

        
# auxiliar function
def change_bm25_parameters(k1, b, index_name, es, num_tries=5):
    
    def close_index():
        es.indices.close(index=index_name)
        
    try_function_n_times(close_index, num_tries)
    
    def update_settings():
        es.indices.put_settings(index=index_name, body={
            "settings": {
                "similarity": {
                  "default": { 
                    "type": "BM25",
                     "b": b,
                     "k1": k1
                    }
                }
            }
        })
        
    try_function_n_times(update_settings, num_tries)
    
    def open_index():
        es.indices.open(index=index_name)
        
    try_function_n_times(open_index, num_tries)
    
    
def plot_tokenizer_word_frequency(tokenizer, min_freq=1, max_freq=20):

    num_valid_words = []

    for i in range(min_freq,max_freq+1):
        print(i,end="\r")
        # most appers at least i times
        tokenizer.update_min_word_frequency(i)
        num_valid_words.append(tokenizer.num_words)
    print()
    
    x = list(range(1,max_freq+1))
    y = num_valid_words

    plt.plot(x,y)

    plt.show()
