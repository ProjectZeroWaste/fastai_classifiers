from fastai.vision import *
from fastai.callbacks import *


from fastai.vision import cutout
from fastai.vision import get_transforms
from fastai.vision import Learner


class enet_x1:
    # training configs
    nruns = 1
    epochs = 1
    cuda_id = 0
    batch_size = 8
    oversample_enable = True
     
    #csiro_c8 web_c8 web_c7 zw_c19_all
    #csv_dataset = '/home/redne/Classifiers/fastai/csiro_prod/ds/ds2_synth100_030121.csv'
    csv_ds = 'zw_c19_all' #'web_s30r70_c4' 'flirkez_s50r50_c4' 'csiro_s30r70_c4' #'csiro_real_easy_c4' #'web_c4' #'flirk_real_c4' #'csiro_real_c4' #'csiro_real_easy_c4' #'ds2_synth100_030121' csiro_s50r50_c4  flirkez_s30r70_c4
    
    csv_dataset = f'/mnt/omreast_users/phhale/csiro_trashnet/datasets/ww_csiro_paper/ds_csv/{csv_ds}.csv'
    
    model = 'efficientnet-b3' #'efficientnet-b3' #'efficientnet-b0' #'efficientnet-b0' 'densenet121' 'resnet50' 'efficientnet-b1 densenet169
    experimentname = f'enetb3_{csv_ds}_oversample_sz300_bs{batch_size}_e{epochs}_032121' #f'dnet121_{csv_ds}_e{epochs}_030321' f'enetb0_{csv_ds}_e{epochs}_030321'
    
    
    #validate Dataset cfg
    #CSIRO VALI
    #val_csv_ds = 'csiro_vidval_c8'
    #MNT_TEST_DIR = '/mnt/omreast_users/phhale/csiro_trashnet/original_samples/'
    #val_csv_dataset = f'/mnt/omreast_users/phhale/csiro_trashnet/datasets/ww_csiro_paper/ds_csv/{val_csv_ds}.csv'

    #Web Val
    val_csv_ds = 'zw_c19_test' # web_val_c7
    MNT_TEST_DIR = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/'
    val_csv_dataset = f'/mnt/omreast_users/phhale/csiro_trashnet/datasets/ww_csiro_paper/ds_csv/{val_csv_ds}.csv'



    # DataBunch
    seed = 42
    #data_train_folder = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/crop_ds0/v2/all_crops/'
    data_train_folder = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/ds2_storm/crop_images/' 
    padding_mode= "reflection"
    #batch_size = 16
    img_input_size = (300, 300) # B3 # r50 b1 = 240 b0 = 244
    #img_input_size = (224, 224)
    #img_input_size = (240, 240)
    #zoom_crop(scale=(0.75, 1.5), do_rand=True)
    xtra_tfms = ([cutout(n_holes=(1, 4), length=(10, 40), p=0.7)]+ [rotate(degrees=(-90,90), p=1)])
    tfms = get_transforms(xtra_tfms=xtra_tfms,
                          max_lighting=0.2,max_warp=0.2,p_affine=0.75,
                          p_lighting=0.75) 
    #tfms = get_transforms()
    
    
    # Learner
    model_save_path = '/mnt/omreast_users/phhale/csiro_trashnet/experiments/classifer_fastai/dev_models/'
    model_pretrain = True

    ## Model Config
    wd=1e-3
    export_learn=True
    lr=3e-3
    div_factor=25
    final_div=1e4