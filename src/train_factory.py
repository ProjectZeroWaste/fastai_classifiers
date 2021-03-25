from fastai.callbacks import OverSamplingCallback, CSVLogger
import numpy as np
import pandas as pd
pd.set_option('precision', 4)
import time
from model_factory import get_learner

EXPERIMENT_RESULT_PATH = '/mnt/omreast_users/phhale/csiro_trashnet/experiments/classifer_fastai/dev_models/experiment_results/'

def train_n_runs(n_runs, epochs, experiment_name, cfg, model_name="efficientnet-b0"):
    """Run training of `model_name` for `n_runs` times"""

    stats_list = []
    for i in range(n_runs):
        print("\n" + "="*70)
        print(f"Training Run #{i+1}")
        print("="*70)
        if i == n_runs-1: export=True
        if model_name in ["efficientnet-b0", "efficientnet-b1","efficientnet-b3", "densenet169", "densenet121","resnet50"]:
            stats_list.append(train(cfg))
        
    # Print history and average of metrics
    df = pd.DataFrame(np.array(stats_list), 
                      columns=["val_loss", "val_acc", "val_error_rate",  "val_top_3", "val_F1",
                               "test_loss", "test_acc", "test_error_rate" , "test_top_3", "test_F1",
                               "time(s)"])

    print("\n" + "="*70)
    print("Metrics history")
    print(df)
    print("")
    print(f"Average metrics over {n_runs} runs")
    print(pd.DataFrame(df.mean(axis=0)).T)
    print(f"Trained model: {model_name} of the last run is exported as "
          f"'{experiment_name}.pkl'")
          
    df.to_csv(f"{EXPERIMENT_RESULT_PATH}results_{model_name}_{experiment_name}.csv")


def train_resnet_classifier(cfg):
    learn = get_learner(cfg)
    # RESNET TRAINING
    epochs = cfg.epochs
    if epochs == None:
        epochs_p1, epochs_p2 = 20, 40
    elif isinstance(epochs, int):
        epochs_p1, epochs_p2 = epochs, epochs
    elif isinstance(epochs, list) and len(epochs) > 1:
        epochs_p1, epochs_p2 = epochs[0], epochs[1]

    # Train
    lr=3e-3
    wd=1e-5

    print("")
    print(f"Training {cfg.model} for {epochs_p1} + {epochs_p2} epochs...")
    start = time.time()
    # Phase 1, train head only
    print("Phase 1, training head...")
    learn.fit_one_cycle(epochs_p1, max_lr=1e-2, wd=1e-5, div_factor=25, 
                        pct_start=0.3,
                        callbacks=[CSVLogger(learn=learn, filename=f'experiment_results/{cfg.experimentname}-history')])
    # Phase 2, train whole model
    learn.unfreeze()
    print("Phase 2, unfreezed and training the whole model...")
    learn.fit_one_cycle(epochs_p2, max_lr=slice(lr/10, lr), wd=wd, 
                        div_factor=25, pct_start=0.3,
                        callbacks=[CSVLogger(learn=learn, filename=f'experiment_results/{cfg.experimentname}-history')])
    
    train_time = time.time() - start
    print("Training completed!")
    print("")   
    return learn, train_time

def train_classifier(cfg):
    learn = get_learner(cfg)
    # Train
    print("")
    print(f"Training {cfg.model} for {cfg.epochs} epochs...")
    start = time.time()
        
    if cfg.oversample_enable:
        learn.fit_one_cycle(cfg.epochs, max_lr=cfg.lr, wd=cfg.wd, div_factor=cfg.div_factor, final_div=cfg.final_div,
                            callbacks=[OverSamplingCallback(learn),
                            CSVLogger(learn=learn, filename=f'experiment_results/{cfg.experimentname}-history')])
    else:
        learn.fit_one_cycle(cfg.epochs, max_lr=cfg.lr, wd=cfg.wd, div_factor=cfg.div_factor, final_div=cfg.final_div,
                            callbacks=[CSVLogger(learn=learn, filename=f'experiment_results/{cfg.experimentname}-history')])

    train_time = time.time() - start
    print("Training completed!")
    print("")   
    return learn, train_time


def train(cfg):

    if cfg.model == 'resnet50':
        learn, train_time = train_resnet_classifier(cfg)
    else:
        learn, train_time = train_classifier(cfg)

    
    val_loss = learn.recorder.val_losses[-1]
    val_acc = learn.recorder.metrics[-1][0]
    val_error_rate = learn.recorder.metrics[-1][1]
    val_top_3 = learn.recorder.metrics[-1][2]
    val_F1 = learn.recorder.metrics[-1][3]
    learn.recorder.plot_metrics(return_fig=True).savefig(f"{EXPERIMENT_RESULT_PATH}metrics_{cfg.model}_{cfg.experimentname}.png")
    print(f"Validation accuracy: \t{val_acc:.05f}")
    if cfg.export_learn:
        learn.export(f"exported_models/{cfg.experimentname}.pkl")

    learn.to_fp32()
    
    # Evaluate test metrics
    from data_utils import get_df_dataset    
    test_data =  get_df_dataset(dataset_path=cfg.MNT_TEST_DIR, df=cfg.val_csv_dataset,
                                    train_mode=False, tfms=None, bs=8,sz=(300, 300),
                                    padding_mode="reflection", print_ds_stats=False)

    # eval for custom DF
    print("")
    print("Evaluating on test set...")
    test_loss, test_acc, test_error_rate, test_top_3, test_F1 = learn.validate(test_data.train_dl)
    print(f"Test accuracy: \t\t{test_acc:.05f}")
    stats = (val_loss, val_acc, val_error_rate, val_top_3, val_F1,
            test_loss, test_acc, test_error_rate, test_top_3, test_F1, 
            train_time)
    
    
    return stats
