from baselines.SSL.arch.PatchPerm import PatchPermAugmentation
from baselines.SSL.arch.SoftCLT_loss import SoftCLT_Loss, SoftCLT_Loss_Sub, SoftCLTLoss_timestep
from baselines.SSL.arch.SoftCLT_spatial_loss import SoftCLT_Spatial_Loss
from baselines.SSL.arch.SoftCLT_temporal_loss import SoftCLTLoss_t, SoftCLTLoss_tod, SoftCLTLoss_global_t
from baselines.SSL.arch.s_contrast import ClusterContrast
from baselines.SSL.arch.t_global_contrast import TGlobalContrast, TGlobalContrast_better


def ssl_factory(**kwargs):
    aug = None
    ssl_name = kwargs['ssl_name']
    if ssl_name == "softclt":
        ssl_module = SoftCLT_Loss(**kwargs)
        aug = PatchPermAugmentation()
    elif ssl_name == "softclt_sub":
        print("softclt_sub")
        ssl_module = SoftCLT_Loss_Sub(**kwargs)
        aug = PatchPermAugmentation()
    elif ssl_name == "softclt_spatial":
        ssl_module = SoftCLT_Spatial_Loss(**kwargs)
        aug = PatchPermAugmentation()
    elif ssl_name == "softclt_temporal":
        ssl_module = SoftCLTLoss_t(**kwargs)
        aug = PatchPermAugmentation()
    elif ssl_name == "softclt_tod":
        ssl_module = SoftCLTLoss_tod(**kwargs)
        aug = PatchPermAugmentation()
    elif ssl_name == "softclt_temporal_global_t":
        ssl_module = SoftCLTLoss_global_t(**kwargs)
        aug = PatchPermAugmentation()
    elif ssl_name == "softclt_timestep":
        ssl_module = SoftCLTLoss_timestep(**kwargs)
        aug = PatchPermAugmentation()
    elif ssl_name == "c_contrast":
        ssl_module = ClusterContrast(**kwargs)
        aug = PatchPermAugmentation()
    elif ssl_name == "t_global_contrast":
        ssl_module = TGlobalContrast(**kwargs)
    elif ssl_name == "t_global_contrast_better":
        ssl_module = TGlobalContrast_better(**kwargs)
    elif ssl_name == "t_global_contrast_better_mapper":
        ssl_module = TGlobalContrast_better(**kwargs)
    else:
        ssl_module = SoftCLT_Loss(similarity_metric="mse", alpha=kwargs["alpha"], tau=kwargs["tau"],
                                       hard=kwargs["hard"])
    ssl_name = kwargs['ssl_name']
    ssl_loss_weight = kwargs['ssl_loss_weight']
    ssl_module.ssl_loss_weight = ssl_loss_weight
    ssl_module.ssl_name = ssl_name
    return ssl_module, aug