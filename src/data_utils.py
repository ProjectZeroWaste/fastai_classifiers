from fastai.vision import ImageDataBunch
from fastai.vision import imagenet_stats
from fastai.vision import crop_pad
import pandas as pd


def get_df_dataset(dataset_path: str = None, df=None, train_mode: bool = True, tfms=None, bs: int = 32, sz=224,
                   padding_mode: str = 'reflection', seed: int = None, split_pct: float = 0.2, print_ds_stats: bool = False):
    df= pd.read_csv(df)
    if train_mode:
        valid_pct = 0.2
    else:
        valid_pct = 0
        tfms = [crop_pad(), crop_pad()]
    data = ImageDataBunch.from_df(path=dataset_path, df=df, valid_pct=valid_pct, seed=42,
                                    fn_col=0,label_col=1, ds_tfms=tfms, size=sz, 
                                    bs=bs,num_workers=4, padding_mode=padding_mode).normalize(imagenet_stats)
    if print_ds_stats:
        show_dataset_stats(data)

    return data


def show_dataset_stats(data):
    print("------ Data Specifications ------")
    print(data)

    print("------ Data Set Specifications ------")
    print("Number of train images:  ", len(data.train_ds.x))
    print("Number of test images :  ", len(data.valid_ds.x))
    print("Number of image folders: ", len(data.classes))
    print(data.classes)
