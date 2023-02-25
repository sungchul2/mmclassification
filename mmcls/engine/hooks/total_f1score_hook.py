from typing import Dict, Optional

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class TotalF1ScoreHook(Hook):
    priority = 'NORMAL'

    def __init__(self, metric: str = 'f1-score'):
        self.metric = metric

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        self._after_epoch(runner, metrics, mode='val')

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        self._after_epoch(runner, metrics, mode='test')

    def _after_epoch(self, runner, metrics, mode: str = 'train') -> None:
        total_score = []
        for k, v in metrics.items():
            if self.metric in k:
                total_score.append(v)
        metrics[f'{mode}/total_average_{self.metric}'] = sum(total_score) / len(total_score)
        runner.message_hub.log_scalars[f'{mode}/total_average_{self.metric}'] = metrics[f'{mode}/total_average_{self.metric}']
