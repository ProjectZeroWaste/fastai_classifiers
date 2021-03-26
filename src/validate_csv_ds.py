#https://stackoverflow.com/questions/56327207/not-able-to-predict-output-in-fastai
#source ~/anaconda3/bin/activate; conda activate fastai
from fastai.vision import *
from fastai.callbacks import *
import numpy as np
import pandas as pd
import time
import os

import warnings
warnings.filterwarnings("ignore")
#defaults.device = torch.device('cpu')

def get_predictions(learn:Learner, imagelist:ImageList):
    learn.data.add_test(imagelist)
    logits, _ = learn.get_preds(ds_type=DatasetType.Test)
    probs = torch.nn.functional.softmax(logits, dim=1)
    max_probs, y_preds = torch.max(probs, dim=1)
    x = imagelist.items
    return x, y_preds.numpy(), max_probs.numpy()

def get_predictions_from_folder(learn:Learner, 
                                test_path:Path) -> (np.ndarray, np.ndarray, np.ndarray):
    """Get predictions of images in a folder
    https://github.com/zerothphase/ai4sea_cv_challenge/blob/master/helper.py
    Parameters:
    -----------
    learn: 
        Inference Learner object
    test_path: 
        Path to the folder where test images are located.
    
    Returns:
    --------
    x:
        Paths of the input images
    y_preds:
        Predicted indices of the class with the highest probability
    max_probs:
        Probabilities of y_preds
    """
    test_imagelist = ImageList.from_folder(test_path)
    x, y_preds, max_probs = get_predictions(learn, test_imagelist)
    return x, y_preds, max_probs

def get_predictions_from_df(learn:Learner, test_df:pd.DataFrame, test_path:Path, 
                            cols:int=0) -> (np.ndarray, np.ndarray, np.ndarray):
    """Get predictions of images from dataframe
    Parameters:
    -----------
    learn: 
        Inference Learner object
    test_df: 
        DataFrame with filenames of the test images in one of the columns.
    test_path: 
        Path to the folder where test images are located.
    cols:
        Column index of the images' filenames.
    
    Returns:
    --------
    x:
        Paths of the input images
    y_preds:
        Predicted indices of the class with the highest probability
    max_probs:
        Probabilities of y_preds
    """
    test_imagelist = ImageList.from_df(test_df, test_path, cols=cols)
    x, y_preds, max_probs = get_predictions(learn, test_imagelist)
    return x, y_preds, max_probs

#test_df:pd.DataFrame = '/home/redne/Classifiers/fastai/csiro_prod/ds/web_val_c4.csv',
#test_df:str ='/home/redne/Classifiers/fastai/csiro_prod/ds/web_val_c4.csv',
def get_df_dataset(learn:Learner, test_df:str ='/home/redne/Classifiers/fastai/csiro_prod/ds/web_val_c4.csv',
                 dataset_path: str = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/') -> (np.ndarray, np.ndarray, np.ndarray):
    from fastai.vision import ImageDataBunch
    from fastai.vision import crop_pad
    from fastai.vision import imagenet_stats
    from fastai.vision import ImageList
    test_df = pd.read_csv(test_df)
    """
    test_imagelist = ImageDataBunch.from_df(path=dataset_path,
                                    df=test_df,
                                    valid_pct=0, seed=42,
                                    fn_col=0,
                                    label_col=1,
                                    ds_tfms=None, size=(300, 300),
                                    padding_mode="reflection").normalize(imagenet_stats)
    """
    test_imagelist = ImageList.from_df(test_df, dataset_path, cols=0)
    x, y_preds, max_probs = get_predictions(learn, test_imagelist)
    return x, y_preds, max_probs
            
def idx_to_classname(y_preds, learn:Learner):
    """Convert predicted indices to classname using mapping from learn."""
    class_preds = np.array(learn.data.single_ds.y.classes)[y_preds]
    return class_preds

def load_default_model():
    #'densenet169_v2_x2_Oversample_112620_stage1.pkl'
    #fastai_efficientnet-b1_120620_x0_all_classes.pkl
    #densenet169_v2_x2_Oversample_112620_stage1.pkl
    #ROOT_PATH = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/crop_ds0/v2/all_crops/'
    
    #FLIKER_ROOT_PATH
    #ROOT_PATH='/mnt/omreast_users/phhale/csiro_trashnet/datasets/crop_ds0/v3_flikr/v1_crops/'
    #ML = densenet161_baseline_taco_120720.pkl
    #learn = load_learner(path=ROOT_PATH+'models/model_export/',
    #                    file='densenet161_csiroflikrV2_x02_122920.pkl')

    ROOT_PATH='/mnt/omreast_users/phhale/csiro_trashnet/experiments/classifer_fastai/dev_models/exported_models/'
    #ML = 'densenet161_csiroflikrV2_x02_122920.pkl'
    #ML = 'raw_scrapers_x00_012321.pkl'
    #ML = 'raw_scrapers_x01_012421.pkl'
    #ML = 'densenet169_scrapersV0_wZW_oversample_x00_012621.pkl'
    #ML = 'densenet169_scrapersV0_vIG_oversample_x00_012821-epoch5.pkl'
    #ML = 'enetB0_scrapV0_oversample_x00_013121-epoch56.pkl'  #'dnet121_scrapV0_020121_9.pkl' #'r50_scrapV0_oversample_x00_013121-epoch14.pkl' #'enetB0_scrapV0_oversample_x00_013121-epoch56.pkl' 
    
    #ML = 'r50_scrapV0_020321_x08_c16-stage2_13.pkl' #x5
    #ML = 'r50_scrapV0_020321_x08_c16-stage2_60.pkl' #x5
    #ML = 'dnet161_scrapV0_020321_x06_e98.pkl' #x6
    #ML = 'enetB0_scrapV0_x06_020321_e54.pkl' #x7

    #ML = 'enetB0_scrapV0_x07_c15_020721_e100.pkl' #x6-2
    #ML = 'r50_scrapV0_x09_c15_020721-stage2_40.pkl' #x9
    #ML = 'experiment-gpus.pkl'
    #ML = 'dnet161_scrapV0v2_x10_c15_021321_e200.pkl' #x12
    #ML = 'enetB3_scrapV0v2_x10_c15_021321_e120.pkl' #x13
    #ML = 'enetB0_scrapV0v2_x10_c15_021321_e134.pkl'
    #ML = 'densnet169_scrapV0_x13_c15_022021.pkl'
    #ML = 'enetB0_scrapV0_x13_c15_022021.pkl' # x14
    #ML = 'densnet121_scrapV0_x13_c14_022021.pkl'
    #ML = 'densnet169_scrapV0_x15_c14_022121.pkl' #x15
    #ML = 'enetB0_scrapV0_x15_c14_022221.pkl' #x16
    #ML =  'enetB0_scrapV0_x14_c14_022021.pkl'
    #ML = 'enetB3_scrapV0_x15_c14_022021.pkl'
    #ML = 'enetB0_web_c4_030121.pkl'
    #ML = 'dnet121_web_c7_noaug_bs8_e20_030621.pkl' #x19
    #ML = 'dnet121_web_c8_e30_030321.pkl' #x20
    #ML = 'dnet121_web_c7_oversample_noaug_sz300_bs8_e20_030721.pkl' #x21
    #ML = 'dnet169_web_c7_oversample_noaug_bs8_e20_030821.pkl' #x22
    #ML = 'r50_web_c7_oversample_bs8_e20_030721.pkl' #x23
    #ML = 'dnet121_csiro_c8_c8_e10_nomix_030321.pkl' #x24
    ML = 'dnet121_zw_c19_all_noaug_oversample_sz300_bs8_e30_031621.pkl' #x25
    
    

    learn = load_learner(path=ROOT_PATH,file=ML)
    print("Loading Model: ", ML)
                         
    learn.to_fp32()
    return learn

def main():
    #source ~/anaconda3/bin/activate; conda activate fastai
    #cd fastia_data_collect/scripts
    # python predict_val_folder.py
    learn = load_default_model()

    #EXP='x19'
    #EXP='x21'
    EXP='x25'

    CSV_DS = 'kaggle_waste_pictures_ds' # zw_c19_test uavvaste_ds_c8 uavvaste_ds_c6_paper csiro_c8 csiro_c7  
    #kaggle_drink_waste_ds kaggle_waste_pictures_ds flirk_real_c6

    SAVE_RESULTS = f'/mnt/omreast_users/phhale/csiro_trashnet/experiments/classifer_fastai/dev_models/prediction_results/{CSV_DS}_preds_{EXP}.csv'
    
    print("="*70)
    print(f"Path to test images folder\t: {str(CSV_DS)}")
    print(f"Path to test images folder\t: {str(learn.data.single_ds.y.classes)}")
    print(f"Batch size\t\t\t: {learn.data.batch_size}")
    print("="*70, "\n\n")

    # Evaluate accuracy on test set
    print("Making predictions...")
    start = time.time()
    #x, y_preds, probs = get_predictions_from_folder(learn, test_path)
    
    # Current
    d_path = f'/mnt/omreast_users/phhale/csiro_trashnet/datasets/ww_csiro_paper/ds_csv/{CSV_DS}.csv'
    test_df = pd.read_csv(d_path)
    #test_df = test_df[test_df.y != 'styrofoam'].reset_index(drop=True) #used for csiro_c8
    x, y_preds, probs = get_predictions_from_df(learn, test_df, test_path='/mnt/omreast_users/phhale/csiro_trashnet/datasets/')
    

    class_preds = idx_to_classname(y_preds, learn)
    output_df = pd.DataFrame(np.array([x, class_preds, probs]).transpose(), 
                            columns=["image_path", "target", "probability"])
    output_df = output_df.sort_values(by=['probability'], ascending=False).reset_index(drop=True)
    print("Showing dataframe of the first 5 predictions")
    print(output_df.head())
    #output_df.to_csv("tools/experiment_results/densenet161_baseline_taco_120720.csv", index=False)
    #output_df.to_csv("tools/experiment_results/results_dontwasteawalk_x1.csv", index=False)
    output_df.to_csv(SAVE_RESULTS, index=False)
    print(f"\n\nAll predictions are exported as {SAVE_RESULTS}")

    infer_time = time.time() - start
    mins, secs = divmod(infer_time, 60)
    print(f"Total inference time: {int(mins)} mins {int(secs)} s ")

if __name__ == "__main__":
    main()
    #BATCH_SIZE = 2
    #learn.data.batch_size = BATCH_SIZE