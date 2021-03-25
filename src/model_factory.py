from fastai.vision import Learner, models
from fastai.vision import cnn_learner, partial
from fastai.vision import LabelSmoothingCrossEntropy
from fastai.vision import accuracy, error_rate
from fastai.metrics import top_k_accuracy, FBeta
from fastai.callback import optim


def get_learner(cfg):
    
    # get training data
    from data_utils import get_df_dataset
    train_databunch =  get_df_dataset(dataset_path='/mnt/omreast_users/phhale/csiro_trashnet/datasets/', df=cfg.csv_dataset,
                                    train_mode=True, tfms=cfg.tfms, bs=cfg.batch_size,
                                    padding_mode=cfg.padding_mode, sz=cfg.img_input_size, print_ds_stats=False)


    top_3_accuracy = partial(top_k_accuracy, k=3)
    METRICS = [accuracy, error_rate, top_3_accuracy, FBeta(beta=1)]


    ## Eff Net
    if cfg.model in ["efficientnet-b0", "efficientnet-b1","efficientnet-b3"]:
        from  efficientnet_util import get_EfficientNet
        eff_net = get_EfficientNet(name=cfg.model, pretrained=cfg.model_pretrain, n_class=train_databunch.c)
        learn = Learner(train_databunch, eff_net, 
                    loss_func=LabelSmoothingCrossEntropy(), 
                    metrics= METRICS, path=cfg.model_save_path).mixup(alpha=0.3)
        learn.to_fp16()
    
    elif cfg.model in ["densenet121","densenet169"]:
        ## Densnet       
        mom = 0.9
        alpha=0.99
        eps=1e-6
        opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
        learn = cnn_learner(train_databunch, models.densenet121, 
                    loss_func=LabelSmoothingCrossEntropy(), 
                    opt_func=opt_func, pretrained=True, wd=1e-2, bn_wd=False, true_wd=True,
                    metrics=METRICS, 
                    path=cfg.model_save_path).mixup(alpha=0.3)
    
    elif cfg.model in ["resnet50"]:
        learn = cnn_learner(train_databunch, models.resnet50, 
                    metrics=METRICS, path=cfg.model_save_path).mixup(alpha=0.3)
                    

    learn.to_fp16()
    return learn