# coding=utf-8

import torch
import json
import numpy as np
import time


class CallBack(object):
    def __init__(self) -> None:
        self.model = None


    def on_train_begin(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    

    



class EarlyStoppingCallback(CallBack):

    def __init__(self, strategy, metric_names, validation_freq, patience, test_data,
                 model_save_path=None,
                 update_callback=None
                #  use_lp, label_prop_func
                 ):
        super().__init__()
        # self.metric_name = metric_name
        
        # self.use_lp = use_lp
        self.strategy = strategy
        self.model_save_path = model_save_path

        # if self.use_lp:
        #     self.label_prop_func = label_prop_func

        if isinstance(metric_names, str):
            metric_names = [metric_names]
        self.val_metric_names = ["val_{}".format(metric_name) for metric_name in metric_names]

        self.validation_freq = validation_freq
        self.patience = patience
        self.patience_counter = 0

        self.min_val_loss = 1000000.0
        self.max_val_score = 0.0


        self.early_stop_logs = None
        self.early_stop_epoch = -1

        self.test_data = test_data

        self.update_callback = update_callback

   

    def on_epoch_end(self, epoch, logs):
        if "val_loss" not in logs:
            return
        

        val_loss = logs["val_loss"]
        # val_score = logs[self.val_metric_name]
        
        val_scores = [logs[val_metric_name] for val_metric_name in self.val_metric_names]
        val_score = np.mean(val_scores)


        stop = False
        if self.strategy == "common":
            reset_patience_counter = val_score > self.max_val_score or val_loss < self.min_val_loss
        elif self.strategy == "loss":
            reset_patience_counter = val_loss < self.min_val_loss
        elif self.strategy == "score":
            reset_patience_counter = val_score > self.max_val_score
        else:
            raise ValueError("Unknown strategy: {}".format(self.strategy))

        # if val_score > self.max_val_score or val_loss < self.min_val_loss:
        if reset_patience_counter:
            self.patience_counter = 0
        else:
            self.patience_counter += self.validation_freq
            if self.patience_counter > self.patience:
                stop = True
                self.model.stop_training = True

        
        if not stop:
            if self.strategy == "common":
                should_update = val_score > self.max_val_score and val_loss < self.min_val_loss
            elif self.strategy == "loss":
                should_update = val_loss < self.min_val_loss
            elif self.strategy == "score":
                should_update = val_score > self.max_val_score
            else:
                raise ValueError("Unknown strategy: {}".format(self.strategy))
            
            # if val_score > self.max_val_score and val_loss < self.min_val_loss:
            if should_update:
            # if True:
                self.early_stop_logs = {
                    "es_{}".format(key): value
                    for key, value in logs.items() if key.startswith("val_")
                }

                
                self.max_val_score = val_score
                self.min_val_loss = val_loss
                self.early_stop_epoch = epoch
                if self.test_data is not None:
                    self.early_stop_logs = {
                        **self.early_stop_logs,
                        **self.model.evaluate(self.test_data, log_prefix="es_eval")
                    }
                
                if self.update_callback is not None:
                    self.update_callback(epoch, logs, self.early_stop_logs, self)

                if self.model_save_path is not None:
                    torch.save(self.model.state_dict(), self.model_save_path)


                # if self.use_lp:
                #     label_prop_logs = self.label_prop_func(model, all_data_loader)
                #     self.early_stop_logs = {
                #         **self.early_stop_logs,
                #         **label_prop_logs
                #     }

                # for key, value in self.early_stop_logs.items():
                #     logs[key] = value

        logs["patience"] = self.patience_counter
        logs["early_stop_epoch"] = self.early_stop_epoch

        if self.early_stop_logs is not None:
            for key, value in self.early_stop_logs.items():
                    logs[key] = value

            




class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)



class LoggingCallback(CallBack):

    def __init__(self, log_path, extra_logs=None):
        super().__init__()

        self.log_path = log_path
        self.extra_logs = extra_logs if extra_logs is not None else {}
        self.start_time = None

    def on_train_begin(self):
        if self.start_time is None:
            self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        
        # if "val_loss" not in logs:
        #     return

        has_eval = False
        for key in logs:
            if key.startswith("es_eval_"):
                has_eval = True
                break

        if not has_eval:
            return

        train_time = time.time() - self.start_time
        
        logs.update({
            **self.extra_logs,
            "epoch": epoch,
            "train_time": train_time
        })

        # if "early_stop_epoch" in self.extra_logs:
        #     early_stop_epoch = self.extra_logs[early_stop_epoch]
        #     if early_stop_epoch == epoch:
        #         pass
        
        if "pre_compute_time" in self.extra_logs:
            logs["all_time"] = self.extra_logs["pre_compute_time"] + train_time

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("{}\n".format(json.dumps(logs, cls=NumpyFloatValuesEncoder)))








import torchvision.utils as vutils
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter



class TensorBoardCallback(CallBack):

    def __init__(self,  log_dir='logs'):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None

    def on_train_begin(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir, flush_secs=1)

    def on_epoch_end(self, epoch, logs=None):
        
        for key, value in logs.items():
            if key  == "epoch":
                continue
            self.writer.add_scalar(key, value, epoch)




