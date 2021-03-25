class base_model_config:
    seed = 42
    model_save_path = '/mnt/omreast_users/phhale/csiro_trashnet/experiments/classifer_fastai/dev_models/'
    model_pretrain = True

    ## Model Config
    wd=1e-3
    export_learn=True
    lr=3e-3
    div_factor=25
    final_div=1e4