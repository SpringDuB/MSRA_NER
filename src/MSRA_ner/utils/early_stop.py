import copy
import logging
from typing import Optional

from torch import nn


class EarlyStopStrategy(object):
    def __init__(self, max_stop_counter):
        super(EarlyStopStrategy, self).__init__()
        self.is_stop = False
        self.best_eval_value = None
        self.best_model_state_dict = None
        self.stop_counter = 0
        self.max_stop_counter = max_stop_counter

    def update(self, eval_value, model: Optional[nn.Module] = None):
        if self.best_eval_value is None:
            self.best_eval_value = eval_value
            self.stop_counter = 0
            if model is not None:
                self.best_model_state_dict = copy.deepcopy(model.state_dict())
        elif eval_value >= self.best_eval_value:
            self.best_eval_value = eval_value
            self.stop_counter = 0
            if model is not None:
                self.best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.stop_counter += 1
            logging.info(f"当前评估指标变差:{self.best_eval_value} -- {eval_value} -- {self.stop_counter}")
            if self.stop_counter > self.max_stop_counter:
                self.is_stop = True

    def stop(self) -> bool:
        return self.is_stop
