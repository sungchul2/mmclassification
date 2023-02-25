from .multi_task_head import MultiTaskHead
from mmcls.registry import MODELS


@MODELS.register_module()
class HierarchMultiTaskHead(MultiTaskHead):...
