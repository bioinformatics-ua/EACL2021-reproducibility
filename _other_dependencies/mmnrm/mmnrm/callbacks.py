import tensorflow as tf
from tensorflow.keras import backend as K

from mmnrm.utils import save_model_weights, load_model_weights, set_random_seed, save_model

import numpy as np

import matplotlib.pyplot as plt

import os
import time
import wandb
import tempfile
import shutil
import math
import h5py

from collections import defaultdict

class Callback:
    def __init__(self, **kwargs):
        pass
    
    def on_epoch_start(self, training_obj, epoch):
        pass
    
    def on_epoch_end(self, training_obj, epoch):
        pass
        
    def on_step_start(self,training_obj, epoch, step):
        pass
        
    def on_step_end(self, training_obj, epoch, step, loss, time):
        pass
    
    def on_train_start(self, training_obj):
        pass
    
    def on_train_end(self, training_obj):
        pass
        
class TriangularLR(Callback):
    """
    From: https://arxiv.org/pdf/1506.01186.pdf
    
    adaptation from: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/cyclical_learning_rate.py
    """
    
    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2020.,
            mode='triangular2',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle'):
        super(TriangularLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.

        self._reset()
        
    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)
        
    def on_step_end(self, training_obj, epoch, step, loss, time):

        self.trn_iterations += 1
        self.clr_iterations += 1
        
        update_lr = self.clr()
        K.set_value(training_obj.optimizer.lr, update_lr)

        
class LRfinder(Callback):
    """
    adapted from: https://gist.github.com/WittmannF/c55ed82d27248d18799e2be324a79473
    """
    def __init__(self, min_lr, max_lr, steps=188, epoch=32, mom=0.9, stop_multiplier=None, 
                 reload_weights=True, batches_lr_update=20, rebuild_model_fn=None):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        if stop_multiplier is None:
            self.stop_multiplier = -20*self.mom/3 + 10 # 4 if mom=0.9
                                                       # 10 if mom=0
        else:
            self.stop_multiplier = stop_multiplier
            
        self.iteration=steps*epoch
        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, \
                                           num=self.iteration//self.batches_lr_update+1)
        
        self.losses = [[]]
        self.avg_losses=[]
        self.discard = batches_lr_update//4
        self.lrs = []
        self.step_count = 0
        
        self.rebuild_model_fn=rebuild_model_fn
        
        self.temp_dir = tempfile.mkdtemp()
    
    def on_train_start(self, training_obj):
        save_model_weights(os.path.join(self.temp_dir,"temp.h5"), training_obj.model)
    
    def on_step_end(self, training_obj, epoch, step, loss, time):
        self.losses[-1].append(loss)
        self.step_count+=1
        
        if self.step_count%self.batches_lr_update==0:

            load_model_weights(os.path.join(self.temp_dir,"temp.h5"), training_obj.model)

            lr = self.learning_rates[self.step_count//self.batches_lr_update]            
            K.set_value(training_obj.optimizer.lr, lr)
            
            avg = sum(self.losses[-1][self.discard:])/len(self.losses[-1][self.discard:])
            if math.isnan(avg):
                avg = 0.7

            self.avg_losses.append(avg)
            self.lrs.append(lr)
            print(lr, avg)
                
            self.losses.append([])
            
        

    def on_train_end(self, training_obj):
        
        load_model_weights(os.path.join(self.temp_dir,"temp.h5"), training_obj.model)
        
        shutil.rmtree(self.temp_dir)
                
        plt.figure(figsize=(12, 6))
        plt.plot(self.lrs, self.avg_losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.show()

        
class Validation(Callback):
    def __init__(self, 
                 validation_collection=None, 
                 test_collection=None, 
                 output_metrics=["recall_100", "map_cut_20","ndcg_cut_20","P_20"],
                 path_store = "/backup/NIR_BioASQ/best_validation_models",
                 interval_val = 1,
                 **kwargs):
        super(Validation, self).__init__(**kwargs)
        self.validation_collection = validation_collection
        self.test_collection = test_collection
        self.current_best = [[ 0 for _m in range(len(output_metrics))] for _ in range(len(validation_collection))]

        self.output_metrics = output_metrics
        self.path_store = path_store
        self.interval_val = interval_val
        self.count = 0
    
    def evaluate(self, model_score_fn, collection):
        generator_Y = collection.generator()
                
        q_scores = defaultdict(list)

        for i, _out in enumerate(generator_Y):
            query_id, Y, docs_ids = _out
            s_time = time.time()
            scores = model_score_fn(Y)[:,0].tolist()
            if not i%50:
                print("\rEvaluation {} | avg 50-time {}".format(i, time.time()-s_time), end="\r")
            q_scores[query_id].extend(list(zip(docs_ids,scores)))

        # sort the rankings
        for query_id in q_scores.keys():
            q_scores[query_id].sort(key=lambda x:-x[1])

        # evaluate
        return collection.evaluate(q_scores)
    
    def on_epoch_start(self, training_obj, epoch):
        if hasattr(self, "model_name"):
            name = self.model_name
        else:
            name = training_obj.model.name 
            
        name += "_val_collection{}_{}"
        self.model_path = os.path.join(self.path_store, name)
    
    def on_epoch_end(self, training_obj, epoch):
        if self.validation_collection is None:
            return None
        
        self.count += 1
        
        if not self.count%self.interval_val:
            metrics = []
            for i,val_collection in enumerate(self.validation_collection):
                
                _metrics = self.evaluate(training_obj.predict_score, val_collection)
                
                _str = "" # use stringbuilder instead
                
                for j,_out_metric in enumerate(self.output_metrics):
                    if _out_metric in _metrics:
                        
                        _str += " | {} {}".format(_out_metric, _metrics[_out_metric])
                        
                        if _metrics[_out_metric]>self.current_best[i][j]:
                            self.current_best[i][j] = _metrics[_out_metric]
                            save_model(self.model_path.format(i,_out_metric), training_obj.model)
                            #save_model_weights(self.model_path.format("map"), training_obj.model)
                            print("Saved current best:", self.current_best[i][j])

                print("Epoch {}{}".format(epoch, _str))
                
                metrics.append((val_collection.name, _metrics))

        else:
            metrics = None
            
        return metrics
    
    def on_train_end(self, training_obj):
        
        # save final model
        # save_model_weights(self.model_path.format("final"), training_obj.model)
        save_model(self.model_path.format("all", "final"), training_obj.model)
        pass
        #if self.test_collection is None:
        #    return None
        
        # restore to the best
        #if self.current_best[0]>0:
        #    load_model_weights(self.model_path, training_obj.model)
        
        #metrics = self.evaluate(training_obj.model_score, self.test_collection)
                  
        #_str = "" # use stringbuilder instead
        #for m in self.output_metrics:
        #    _str += " | {} {}".format(m, metrics[m])

        #print("\nTestSet final evaluation{}".format(epoch, _str))

        #return metrics

class PrinterEpoch(Callback):
    def __init__(self, steps_per_epoch=None, **kwargs):
        super(PrinterEpoch, self).__init__(**kwargs)
        self.losses_per_epoch = []
        self.steps_per_epoch = steps_per_epoch
    
    def on_epoch_start(self, training_obj, epoch):
        self.losses_per_epoch.append([])
    
    def on_epoch_end(self, training_obj, epoch):
        avg_loss = sum(self.losses_per_epoch[-1])/len(self.losses_per_epoch[-1])
        print("Epoch {} | avg Loss {}".format(epoch, avg_loss))
        return avg_loss
        
    def on_step_end(self, training_obj, epoch, step, loss, time):
        self.losses_per_epoch[-1].append(loss)
        print("\rStep {}/{} | Loss {} | time {}".format(step, self.steps_per_epoch, loss, time), end="\r")
    
class WandBValidationLogger(Validation, PrinterEpoch):
    def __init__(self, wandb_args, **kwargs):
        self.wandb = wandb
        self.wandb.init(**wandb_args)
        self.wandb.run.save()
        self.model_name = self.wandb.run.name
        
        Validation.__init__(self, **kwargs)
        PrinterEpoch.__init__(self, **kwargs)
        
        
    def on_epoch_start(self, training_obj, epoch):
        PrinterEpoch.on_epoch_start(self, training_obj, epoch)
        Validation.on_epoch_start(self, training_obj, epoch)
    
    def on_epoch_end(self, training_obj, epoch):
        avg_loss = PrinterEpoch.on_epoch_end(self, training_obj, epoch)
        collection_metrics = Validation.on_epoch_end(self, training_obj, epoch)
        if collection_metrics is not None:
            _log = {'loss': avg_loss, 'epoch': epoch}
            
            for collection_name, metrics in collection_metrics:
                
                for _out_metric in self.output_metrics:
                    if _out_metric in metrics:
                        _log[collection_name+"_"+_out_metric] = metrics[_out_metric]
                    
                    #self.wandb.run.summary["best_"+collection_name+"_"+m] = self.current_best[i]
              
            self.wandb.log(_log)
        
            
        else:
            self.wandb.log({'loss': avg_loss,
                           'epoch': epoch})
        
    def on_step_end(self, training_obj, epoch, step, loss, time):
        PrinterEpoch.on_step_end(self, training_obj, epoch, step, loss, time)
        self.wandb.log({'loss': float(loss)})
        
    def on_train_end(self, training_obj):
        Validation.on_train_end(self, training_obj) # just do a save
        
        #metrics = Validation.on_train_end(self, training_obj)
        #if metrics is not None:
        #    self.wandb.run.summary["test_"+self.output_metrics[0]] = metrics[self.output_metrics[0]]
        #    self.wandb.run.summary["test_"+self.output_metrics[1]] = metrics[self.output_metrics[1]]
        
def step_decay(epoch, initial_lrate):

    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    return lrate


class LearningRateScheduler(Callback):
    def __init__(self, initial_learning_rate, lr_fn=step_decay, **kwargs):
        super(LearningRateScheduler, self).__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.lr_fn = lr_fn
        
    def on_epoch_end(self, training_obj, epoch):
        
        new_lr = self.lr_fn(epoch, self.initial_learning_rate)

        # update the lr
        K.set_value(training_obj.optimizer.lr, new_lr)

    

