import os
from typing import Tuple

import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


class MyDataSet(Dataset):
    def __init__(self, normalized: bool = False) -> None:
        """
        用於訓練的預測訓練時間的Dataset
        Args:
            sequence_length: 如果需要時間序列數據，需要設為非1的自然數
        """
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        data = pl.read_csv(os.path.join(self.current_path, "../../dataset/dataset.csv"))
        data = self.__balance(data)
        features = data.select(
            [
                "cpu_count",
                "cpu_freq_max",
                "cpu_freq_min",
                "gpu_core",
                "gpu_freq_max",
                "gpu_freq_min",
                "gpu_mem",
                "mem",
                "parameters",
            ]
        ).to_numpy()
        labels = data["time"].to_numpy()

        self.features = torch.tensor(np.array(features, dtype=np.float32))
        self.labels = torch.tensor(np.array(labels, dtype=np.float32))
        if normalized:
            self.__normalize()

    def __normalize(self) -> None:
        mean = torch.mean(self.features, dim=(0, 1))
        std = torch.std(self.features, dim=(0, 1))
        std[std == 0] = 1
        self.features = (self.features - mean) / std
        mean = torch.mean(self.labels)
        std = torch.std(self.labels)
        std[std == 0] = 1
        self.labels = (self.labels - mean) / std
        os.makedirs(os.path.join(self.current_path, "../../info"), exist_ok=True)
        info = {
            "mean": mean,
            "std": std,
        }
        df = pl.DataFrame(info)
        df.write_csv(os.path.join(self.current_path, "../../info/info.csv"))

    def __balance(self, data: pl.DataFrame) -> pl.DataFrame:
        # 計算每個 gpu_core 的出現次數
        count_df = data.group_by("gpu_core").agg(pl.count().alias("count"))
        # print(count_df)
        if count_df["count"].sort()[0] == count_df["count"].sort()[1]:
            return data
        min_count = count_df["count"].sort()[0]

        # 建立需要 explode 的欄位清單：排除 group key "gpu_core"
        agg_columns = [col for col in data.columns if col != "gpu_core" or col == ""]

        balanced_data = (
            data.group_by("gpu_core")
            .agg([pl.all().sample(n=min_count)])  # 每個 group 的欄位變成 List
            .explode(agg_columns)  # 僅對聚合後的 List 欄位展開
        )
        count_df = balanced_data.group_by("gpu_core").agg(pl.count().alias("count"))
        print(count_df)
        balanced_data.write_csv(
            os.path.join(self.current_path, "../../dataset/dataset.csv")
        )
        return balanced_data

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self.labels[index], self.features[index]


def get_data_loader(
    batch_size: int = 32,
    num_workers: int = 0,
    normalized: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
        回傳三個dataloader
    Args:
        batch_size: 批次大小
        num_workers: cpu用幾個core
        sequence_length: 如果需要時間序列數據，需要設為非1的自然數
    Returns:
        train_loader, vali_loader, test_loader
    """
    dataset = MyDataSet(normalized=normalized)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    vali_size = int(0.2 * train_size)
    train_size -= vali_size
    train_set, vali_set, test_set = random_split(
        dataset, [train_size, vali_size, test_size]
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    vali_loader = DataLoader(
        vali_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, vali_loader, test_loader


if __name__ == "__main__":
    dataset = MyDataSet(normalized=True)
    print(dataset[0])
    dataset = MyDataSet(normalized=False)
    print(dataset[0])
    _, _, _ = get_data_loader(normalized=True)
