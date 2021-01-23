"""
Auxiliar code needed to better handle text inputs
"""
from collections import defaultdict
from os.path import join
import random



def TREC_queries_transform(queires, number_parameter="number", fn = lambda x:x["title"]):
    return list(map(lambda x:{"id":x[number_parameter],
                         "query":fn(x)
                        }, queires))

def TREC_goldstandard_transform(goldstandard):
    _g = {}
    for _id, relevance in goldstandard.items():
        _g[_id]=defaultdict(list)
        
        for doc in relevance:
            _rel = int(doc[1])
            
            if _rel<0: #assert
                raise RuntimeError("value of relevance is negative??", doc[1])
            
            _g[_id][_rel].append(doc[0])
            
    return _g

# fn should be identity: "lambda x:x" (pls replace this default function)
def TREC_results_transform(results, fn = lambda x:{"id":x["DOCNO"], "text":x["HEADER"]+" "+x["HEADLINE"]+" "+x["TEXT"]}):
    _g = {}
    for _id, relevance in results.items():
        _g[_id] = list(map(fn, relevance))
    return _g

def TREC04_merge_goldstandard_files(list_of_files, path_to_store):
    
    name = join(path_to_store, "temp_gs_{}.txt".format(int(random.random()*10000)))
    with open(name,"w") as wf:
        for f in list_of_files:
            with open(f,"r") as rf:
                for line in rf:
                    wf.write(rf)
                    
    return name
   
# backwards compatibility
TREC04_queries_transform = TREC_queries_transform
TREC04_goldstandard_transform = TREC_goldstandard_transform
TREC04_results_transform = TREC_results_transform
 