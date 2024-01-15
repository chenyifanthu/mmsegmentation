from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class BedsheetDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('background', 'edge'),
        palette=[[0, 0, 0], [255, 0, 0]])

    def __init__(self, data_root: str, split: str, **kwargs):
        assert split in ['train', 'val']            
        super().__init__(data_root=data_root, 
                         data_prefix=dict(img_path=f'img_dir/{split}', seg_map_path=f'ann_dir/{split}'),
                         **kwargs)