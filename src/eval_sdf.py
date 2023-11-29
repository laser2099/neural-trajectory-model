import torch

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from env_dataset import SDFDataset
from model_utils import get_model_device
from run_models import run_inference
from eval_utils import hist_plot

def eval_sdf_model(
    model, dataset, 
    criterion= nn.L1Loss(),
    batch_size = 1000,
    num_workers=8,
):
    if len(dataset) < 0:
        print("eval dataset is empty")
        return None
    device = get_model_device(model)
    test_loader = DataLoader(dataset,
                        shuffle=False,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=False,
                        )
    model.eval()
    eval_loss = []
    for i,data in tqdm(enumerate(test_loader)):
        points, sdf_vs = data
        points = points.float().to(device)
        sdf_vs = sdf_vs.float().to(device)    
        with torch.no_grad():
            output = model(points)
            loss = criterion(output, sdf_vs)
            eval_loss.append(loss.item())
    return np.array(eval_loss).mean()


class ValidFunction:
    def __init__(
        self,
        eval_data_path,
        criterion= nn.L1Loss(),
        batch_size = 1000,
        num_workers=8,
    ):
        self.num_workers = num_workers
        self.criterion = criterion
        self.batch_size = batch_size
        self.datasets = {
            "overall":  SDFDataset(eval_data_path, ),
            "uniform": SDFDataset(eval_data_path, ["uniform"]),
            "inside": SDFDataset(eval_data_path, ["insides"]),
            "surface_points": SDFDataset(eval_data_path, ["surface_points"]), 
            "near_surface_points": SDFDataset(eval_data_path, ["near_surface_points"]), 
        }
        print("eval data sizes: ", {k: len(d) for k, d in self.datasets.items()})
    
    def __call__(self, model):
        metrics = {
            k: eval_sdf_model(model, dataset, self.criterion, self.batch_size, self.num_workers)
            for k, dataset in self.datasets.items()
        }
        return metrics



def get_env_sdf(
    sdf_model,
    env_samples,
    batch_size=128,
):
    ret = {
        k:  run_inference(
            sdf_model, 
            x_in=env_samples[k].astype('float32'), 
            batch_size=batch_size)
        for k, v in env_samples.items() if "_sdf" not in k
    }

    print("{} instances using total computing time: {}".format(
            sum( [v['output'].shape[0] for _, v in ret.items()]), 
            sum( [v['total_compute_time'] for _, v in ret.items()] ),
        )
    )
    ret = {
        k: v["output"]
        for k, v in ret.items()
    }
    return ret


def eval_sdf(env_samples, pred_sdfs, truth_sdfs):
    from sklearn.metrics import r2_score
    import pandas as pd
    ret = []
    for k in env_samples:
        if "_sdf" in k:
            continue
        coordindates = env_samples[k].astype('float32')
        df = pd.DataFrame(
            {
            "x": coordindates[:, 0],
            "y": coordindates[:, 1],
            "z": coordindates[:, 2],
            "predict": pred_sdfs[k],
            "truth": truth_sdfs[k],
            }
        )
        df[k + "_rel_abs"] = ((df['truth'] - df['predict']) / df['truth']).abs()
        df[k + "_abs"] = ((df['truth'] - df['predict'])).abs()
        print(k + "_r2: ", r2_score(df['truth'], df['predict']))
    #     print(k)
        print(df.describe())
#         df['predict'].hist(label="predict")
#         df['truth'].hist(label="truth")
#         plt.legend()
#         plt.show()
        ret.append( (k, df))
    return ret


    
def eval_sdf(env_samples, pred_sdfs, truth_sdfs, hist_bins=10):
    import matplotlib.pyplot as plt
    epsilon = 1e-5
    from sklearn.metrics import r2_score, mean_absolute_percentage_error
    import pandas as pd
    ret = []
    for k in env_samples:
        if "_sdf" in k:
            continue
        coordindates = env_samples[k].astype('float32')
        df = pd.DataFrame(
            {
            "x": coordindates[:, 0],
            "y": coordindates[:, 1],
            "z": coordindates[:, 2],
            "predict": pred_sdfs[k],
            "truth": truth_sdfs[k],
            }
        )
        df[k + "_rel_abs"] = ((df['truth'] - df['predict']) / (1e-5 + df['truth'].abs())).abs()
        df[k + "_abs"] = ((df['truth'] - df['predict'])).abs()
        print(k + "_r2: ", r2_score(df['truth'] + epsilon, df['predict'] + epsilon))
        print(k + "_mape: ", mean_absolute_percentage_error(df['truth'] + epsilon, df['predict'] + epsilon))
        
    #     print(k)
        print(df.describe())
#         df['truth'].hist(label="truth", bins=hist_bins)
#         df['predict'].hist(label="predict", bins=hist_bins)
#         plt.legend()
        hist_plot({"truth": df["truth"], "predict": df['predict']}, bins=100)

        plt.show()
        hist_plot({k + "_abs": df[k + "_abs"]}, bins=100)
        plt.show()


        ret.append( (k, df))
    return ret