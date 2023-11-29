import numpy as np
from torch.utils.data import Dataset


def straight_line(start_ends, num_points):
    # start_ends: bsz x num_agents x 2 x 4
    starts = start_ends[:, :, :1, :]
    ends = start_ends[:, :, 1:, :]
    step = (ends - starts) / (num_points - 1)
    line_range = np.arange(num_points).reshape(1, 1, -1, 1)
    lines = starts + line_range * step
    return lines


class TrajDataset(Dataset):
    def __init__(self, path, start_end_line_feed=True):
        data = np.load(path, allow_pickle=True)[()]
        self.start_ends = data['start_ends'] # shape: data_size x num_agents x 2 x 4
        self.gt_trajs = data['gt_trajs'] #  shape: data_size x num_agents x num_points x 4
        _, self.num_agents, self.num_points, _ = self.gt_trajs.shape
        self.start_end_line_feed = start_end_line_feed


    def __len__(self) -> int:
        return len(self.start_ends)

    def __getitem__(self, index):
        start_end = self.start_ends[index]
        line = straight_line(np.expand_dims(start_end, 0), self.num_points)[0]
        x = line if self.start_end_line_feed else start_end
        return x, self.gt_trajs[index]


def test_dataset():
    from torch.utils.data import DataLoader

    start = np.random.random((1, 8, 4))
    end = np.random.random((1, 8, 4))
    start_ends = np.stack( [start.reshape(1, 8, 4), end.reshape(1, 8, 4)], axis=2)
    start_ends = np.vstack([start_ends, start_ends])
    print(start_ends.shape)
    line = straight_line(start_ends, 128)


    print(line.shape)
    np.allclose(start_ends[:, :, 0, :], line[:, :, 0, :]), np.allclose(start_ends[:, :, 1, :], line[:, :, -1, :])
    [
        np.allclose(
            line[:, :, i + 1, :] - line[:, :, i, :],
            line[:, :, i + 2, :] - line[:, :, i + 1, :],
        )
        for i in range(126)
    ]


    data_path = '/Users/tang/workspace/experiments/multi_nt/training_trajs/forest_eight_test_final'
    traj_data_path = "/Users/tang/workspace/data/neural_traj_data/traj_dataset/eval_trajs_8agents.npy"
    dataset = convert_dataset(data_path, traj_data_path)


    d = TrajDataset(traj_data_path)
    start_end, trajs = d[0]
    start_end.shape, trajs.shape
    batch_size = 128
    num_workers = 0
    dataloader =  DataLoader(d,
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=False
                                )

    for i, (lines, trajs) in enumerate(dataloader):
        print(i, lines.shape, trajs.shape)


def convert_dataset(old_dataset_path, save_to=None):
    from raw_nt_dataset import nt_dataset as multi_nt_dataset
    num_points = 128
    dataset = multi_nt_dataset(old_dataset_path, num_points)
    
    start_ends = np.stack(
        [
            np.stack( [dataset[i]['start'], dataset[i]['end']], axis=-2) 
            for i in range(len(dataset))
        ],
        axis=0
    )
    gt_trajs = np.stack(
        [
            dataset[i]['gt_traj']
            for i in range(len(dataset))
        ],
        axis=0
    )
    ret = dict(
        start_ends=start_ends, 
        gt_trajs=gt_trajs
    )
    if save_to is not None:
        np.save(
            save_to,
            ret
        )
        print('saving traj data to: ', save_to)
    return ret



def convert_datasets():
    old_train_path = "/home/dell/zihan/multi_nt/training_trajs/eight_test_final"
    train_traj_data_path = "/home/dell/zihan/open_source_code/forest_eight/src/test_trajs_8agents.npy"
    dataset = convert_dataset(old_train_path, train_traj_data_path)
    print(dataset['start_ends'].shape, dataset['gt_trajs'].shape)


if __name__=="__main__":
    convert_datasets()