import tempfile
import shutil
import os 
import subprocess

def f_recall(predictions, expectations, at=500):
    """
    predictions: list of list of pmid
    expectations: list of list of pmid
    """

    assert len(predictions) == len(expectations)

    return sum([__recall(predictions[i][:at], expectations[i]) for i in range(len(predictions))])/len(predictions)


def __recall(prediction, expectation):
    """
    prediction: list of cut at-k pmid each element should be a tuple (pmid,score) or (pmid)
    expectation: list of valid pmid

    return recall value
    """

    # solve the indetermination but THIS IS STUPID ITS A DATASET ERROR
    if len(expectation) == 0:
        return 0

    return sum([(1 if pmid in expectation else 0) for pmid in prediction])/len(expectation)


def __precision(prediction, expectation):
    """
    prediction: list cut at-k pmid, each element should be a tuple (pmid,score) or (pmid)
    expectation: list of valid pmid
    return precision
    """
    if len(prediction)==0:
        return 0 # assume 0

    return sum([ 1 if pmid in expectation else 0 for pmid in prediction])/len(prediction)


def __average_precision_at(prediction, expectation, bioASQ_version, use_len=False, at=10):
    """
    predictions: list of pmid, each element can be a tuple (pmid,score) or (pmid)
    expectations: list of valid pmid
    return average precision at k
    """

    #assert len(prediction)>0

    binary_relevance = [ 1 if pmid in expectation else 0 for pmid in prediction[:at] ]
    precision_at = [ __precision(prediction[:i],expectation) for i in range(1,at+1) ]

    if bioASQ_version==8:
        if len(expectation)==0:
            return 0
        return sum([a*b for a,b in zip(precision_at,binary_relevance)])/min(len(expectation),10)
    elif bioASQ_version==7:
        return sum([a*b for a,b in zip(precision_at,binary_relevance)])/10
    elif use_len:
        return sum([a*b for a,b in zip(precision_at,binary_relevance)])/len(expectation)
    elif sum(binary_relevance)>0:
        return sum([a*b for a,b in zip(precision_at,binary_relevance)])/sum(binary_relevance)
    else: #The indetermination 0/0 will be consider 0
        return 0

def f_map(predictions, expectations, at=10,bioASQ_version = None, use_len=False ):
    """
    predictions: list of list of pmid
    expectations: list of list of pmid
    
    bioASQ_version: 7 - bioasq 7b
                    8 - bioasq 8b
    """
    assert len(predictions) == len(expectations)

    return sum([ __average_precision_at(predictions[j],expectations[j],bioASQ_version,use_len, at) for j in range(len(predictions))])/len(predictions)


class Evaluator():
        
    def get_config(self):
        data_json = {
            "class": self.__class__
        } 
        
        return data_json
    
    def save(self, path):
        with open(path+".p", "wb") as f:
            pickle.dump(self.get_config(), f)
            
    @classmethod
    def load(cls, **config):
        
        return cls(**config)
    
    def evaluate(self, ranked_query_docs):
        raise NotImplementedError("This method must be overridden")
        
class BioASQ_Evaluator(Evaluator):
    def __init__(self,
                 goldstandard,
                 bioasq_version=8,
                 **kwargs):
        super(Evaluator, self).__init__()
        self.goldstandard = goldstandard
        self.gs_cached = None
        self.bioasq_version = bioasq_version
        
    def get_config(self):
        super_config = super().get_config()
        
        data_json = {
            "goldstandard": self.goldstandard,
            "bioasq_version": self.bioasq_version
        }
        
        return dict(data_json, **super_config) #fast dict merge
    
    def _prepere_goldstandard(self):
        if self.gs_cached is None:
            self.gs_cached = dict(map(lambda x:(x["id"], x["documents"]), self.goldstandard))
            
        return self.gs_cached
    
    def evaluate(self, ranked_query_docs):
        metrics = {}
        gs = self._prepere_goldstandard()
        
        prediction = []
        expectation = []
        
        for i, rank_data in enumerate(ranked_query_docs.items()): 
            prediction.append(list(map(lambda x:x[0], rank_data[1])))
            expectation.append(gs[rank_data[0]])
            
        metrics["map@10"] = f_map(prediction, expectation, bioASQ_version=self.bioasq_version)
        metrics["recall@10"] = f_recall(prediction, expectation, at=10)
        
        return metrics

    def evaluate_oracle(self):
        metrics = {}
        gs = self._prepere_goldstandard()
        
        prediction = []
        expectation = []
        
        for rank_data in gs.values(): 
            prediction.append(rank_data)
            expectation.append(rank_data)
            
        metrics["map@10 bioASQ"] = f_map(prediction, expectation, bioASQ_version=self.bioasq_version)
        metrics["map@10"] = f_map(prediction, expectation)
        metrics["recall@10"] = f_recall(prediction, expectation, at=10)
        
        return metrics
    
class TREC_Evaluator(Evaluator):
    def __init__(self,
                 goldstandard_trec_file,
                 trec_script_eval_path,
                 **kwargs):
        super(Evaluator, self).__init__()
        self.goldstandard_trec_file = goldstandard_trec_file 
        self.trec_script_eval_path = trec_script_eval_path
        
    def get_config(self):
        super_config = super().get_config()
        
        data_json = {
            "goldstandard_trec_file": self.goldstandard_trec_file,
            "trec_script_eval_path": self.trec_script_eval_path,
        }
        
        return dict(data_json, **super_config) #fast dict merge
    
    def __metrics_to_dict(self, metrics):
        return dict(map(lambda k:(k[0], float(k[1])), map(lambda x: tuple(map(lambda y: y.strip(), x.split("\tall"))), metrics.split("\n")[1:-1])))
    
    def evaluate(self, ranked_query_docs):
        metrics = None
        temp_dir = tempfile.mkdtemp()
        
        try:
            with open(os.path.join(temp_dir, "qret.txt"), "w") as f:
                for i, rank_data in enumerate(ranked_query_docs.items()):
                    for j,doc in enumerate(rank_data[1]):
                        _str = "{} Q0 {} {} {} run\n".format(rank_data[0],
                                                           doc[0],
                                                           j+1,
                                                           doc[1])
                        f.write(_str)
                        
            # evaluate
            trec_eval_res = subprocess.Popen(
                [self.trec_script_eval_path, '-c' ,'-M1000','-m', 'all_trec', self.goldstandard_trec_file, os.path.join(temp_dir, "qret.txt")],
                stdout=subprocess.PIPE, shell=False)

            (out, err) = trec_eval_res.communicate()
            trec_eval_res = out.decode("utf-8")
            #print("trec_result")
            #print(trec_eval_res)
            
            metrics = self.__metrics_to_dict(trec_eval_res)

        except Exception as e:
            raise e # maybe handle the exception in the future
        finally:
            # always remove the temp directory
            print("Remove {}".format(temp_dir))
            shutil.rmtree(temp_dir)
            
        return metrics

TREC_Robust04_Evaluator = TREC_Evaluator