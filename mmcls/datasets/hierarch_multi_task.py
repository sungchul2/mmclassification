import json
import os
from collections import defaultdict
from typing import List, Union

import pandas as pd
from mmcls.datasets import MultiTaskDataset
from mmcls.registry import DATASETS


@DATASETS.register_module()
class HierarchMultiTaskDataset(MultiTaskDataset):

    def __init__(self, seed: Union[int, List[int]], labels_map: str, *args, **kwargs):
        self.seed = seed
        self._labels_map = json.load(open(labels_map, 'r'))
        self._labels_attr = ['POA_attribution', 'activity_category', 'activity_type']

        super().__init__(*args, **kwargs)

    def load_data_list(self, ann_file, metainfo_override=None) -> List[dict]:
        if isinstance(self.seed, (tuple, list)):
            _data_list = None
            for seed in self.seed:
                if _data_list is None:
                    _data_list = pd.read_csv(os.path.join(ann_file.format(seed)))
                else:
                    _next_seed = pd.read_csv(os.path.join(ann_file.format(seed)))
                    _data_list = pd.concat((_data_list, _next_seed))
        else:
            _data_list = pd.read_csv(os.path.join(ann_file.format(self.seed)))

        for _labels_attr in self._labels_attr:
            _data_list[_labels_attr] = _data_list[_labels_attr].apply(lambda x: self._labels_map[_labels_attr+'_map'][x])

        _results = []
        for i in range(len(_data_list)):
            _item = defaultdict(dict)
            _labels = {}
            for k, v in _data_list.iloc[i].items():
                if k == 'image_name':
                    _item['img_path'] = os.path.join(self.data_root, v)
                elif k in self._labels_attr:
                    _labels[k] = v
            _item['gt_label'].update(self.divide_task(_labels))
            _results.append(_item)

        return _results

    def divide_task(self, labels):
        """Temp function to divide original 3 tasks to temp 5 tasks."""
        _results = {}
        if labels[self._labels_attr[2]] == 0:
            # NonPartner.com
            _results.update({'task1': 0})
        elif labels[self._labels_attr[2]] == 1:
            # Member.com
            _results.update({'task1': 1, 'task2': 0})
        elif labels[self._labels_attr[2]] == 2:
            # Online Display
            _results.update({'task1': 1, 'task2': 1})
        elif labels[self._labels_attr[2]] == 3:
            # Magazine/Newspaper
            _results.update({'task1': 2, 'task3': 0})
        elif labels[self._labels_attr[2]] == 4:
            # Billboard/Transit
            _results.update({'task1': 2, 'task3': 1})
            if labels[self._labels_attr[1]] == 3:
                _results.update({'task5': 0}) # Out of Home
            else:
                _results.update({'task5': 1}) # Out of Home Media
        elif labels[self._labels_attr[2]] == 5:
            # Collateral
            _results.update({'task1': 2, 'task3': 2})
        elif labels[self._labels_attr[2]] == 6:
            # Misc
            _results.update({'task1': 2, 'task3': 3})
            if labels[self._labels_attr[1]] == 0:
                _results.update({'task4': 0}) # Digital Media
            else:
                _results.update({'task4': 1}) # Paid Social Media
        elif labels[self._labels_attr[2]] == 7:
            # IndustryPartner.com
            _results.update({'task1': 1, 'task2': 2})

        return _results


if __name__ == '__main__':
    dataset = HierarchMultiTaskDataset(
        seed=1,
        labels_map='./data/labels_map.json',
        ann_file='../samples/ref_id_1_0.5_2/cv_{}.csv',
        data_root='./data/TRAIN_IMAGES')

    data_list = dataset.load_data_list()
    print('seed : int')
    print('-> data_list :', len(data_list))

    dataset = HierarchMultiTaskDataset(
        seed=[1, 2, 3, 4, 5],
        labels_map='./data/labels_map.json',
        ann_file='../samples/ref_id_1_0.5_2/cv_{}.csv',
        data_root='./data/TRAIN_IMAGES')

    data_list = dataset.load_data_list()
    print('seed : list')
    print('-> data_list :', len(data_list))
