import os
from easydict import EasyDict
# architecture 
from basicts.archs.Stat_arch import AutoRegressive, VectorAutoRegression
# runner
from basicts.runners.Stat_runner import StatRunner
from basicts.data.base_dataset import BaseDataset
from basicts.metrics.mae import masked_mae
from basicts.metrics.mape import masked_mape
from basicts.metrics.rmse import masked_rmse
from basicts.losses.losses import maksed_l1_loss

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = 'Stat model configuration'
CFG.RUNNER  = StatRunner
CFG.DATASET_CLS   = BaseDataset
CFG.DATASET_NAME  = "PEMS07"
CFG.DATASET_TYPE  = 'Traffic speed'
CFG.GPU_NUM = 1
CFG.SEED    = 1
CFG.CUDNN_ENABLED = True
CFG.METRICS = {
    "MAE": masked_mae,
    "RMSE": masked_rmse,
    "MAPE": masked_mape
}

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME  = 'Stat Model'
CFG.MODEL.ARCH  = VectorAutoRegression
CFG.MODEL.PARAM = {
    'p': 12,
    'input_length': 12,
    'output_length': 12,
    'num_time_series': 883
}
CFG.MODEL.FROWARD_FEATURES = [0, 1]            # traffic speed, time in day
CFG.MODEL.TARGET_FEATURES  = [0]                # traffic speed

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = maksed_l1_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.001,
    "weight_decay":1.0e-5,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[50],
    "gamma":0.1
}

# ================= train ================= #
CFG.TRAIN.CLIP       = 5
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA          = EasyDict()
CFG.TRAIN.NULL_VAL      = 0.0
## read data
CFG.TRAIN.DATA.DIR      = 'datasets/' + CFG.DATASET_NAME
## dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE   = 64
CFG.TRAIN.DATA.PREFETCH     = False
CFG.TRAIN.DATA.SHUFFLE      = True
CFG.TRAIN.DATA.NUM_WORKERS  = 2
CFG.TRAIN.DATA.PIN_MEMORY   = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
## read data
CFG.VAL.DATA.DIR      = 'datasets/' + CFG.DATASET_NAME
## dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE     = 64
CFG.VAL.DATA.PREFETCH       = False
CFG.VAL.DATA.SHUFFLE        = False
CFG.VAL.DATA.NUM_WORKERS    = 2
CFG.VAL.DATA.PIN_MEMORY     = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# validating data
CFG.TEST.DATA = EasyDict()
## read data
CFG.TEST.DATA.DIR      = 'datasets/' + CFG.DATASET_NAME
## dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE    = 64
CFG.TEST.DATA.PREFETCH      = False
CFG.TEST.DATA.SHUFFLE       = False
CFG.TEST.DATA.NUM_WORKERS   = 2
CFG.TEST.DATA.PIN_MEMORY    = False
