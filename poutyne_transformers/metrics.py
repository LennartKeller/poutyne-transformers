from typing import Any, Callable, Dict
#from poutyne.framework.metrics import get_names_of_metric


class MetricWrapper:
    def __init__(self, metric: Callable, pred_key: str = "logits", y_key: str = None):
        self.metric = metric
        self.pred_key = pred_key
        self.y_key = y_key
        self._set_metric_name(metric)

    def _set_metric_name(self, metric):
        #metric, name = get_names_of_metric(metric)
        self.__name__ = metric.__name__

    def __call__(self, outputs: Dict[str, Any], y_true: Any):
        y_pred = outputs[self.pred_key]
        if self.y_key is not None:
            y_true = outputs[self.y_key]
        return self.metric(y_pred, y_true)
