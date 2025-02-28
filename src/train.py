import argparse
import os

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from rich.progress import BarColumn, Progress, TimeRemainingColumn, track
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import core


class Trainer:
    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 0.01,
        epochs: int = 30,
        num_workers: int = 0,
        normalized: bool = False,
        save_path: str = "",
        info_path: str = "",
    ) -> None:
        """
        產生Model的Trainer

        Args:
            batch_size: 批次大小
            lr: 學習率
            epochs: 訓練的循環次數
            num_workers: dataloader的worker
            sequence_length: 資料的時間序列長度
            save_path: model的儲存位置，如果是空字串則不儲存
        """
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_workers = num_workers
        self.normalized = normalized
        self.save_path = save_path
        current_path = os.path.dirname(os.path.abspath(__file__))
        if self.save_path:
            self.save_path = os.path.join(current_path, "..", self.save_path)
            os.makedirs(self.save_path, exist_ok=True)
            self.save_path = os.path.join(self.save_path, "model.pth")
            self.save_flag = True
        self.mean = 0
        self.std = 1
        if info_path:
            info_path = os.path.join(current_path, "..", info_path, "info.csv")
            if os.path.exists(info_path):
                df = pl.read_csv(info_path)
                self.mean = df["mean"].to_list()[0]
                self.std = df["std"].to_list()[0]
                del df
        self.save_flag = False
        self.__pre_environment()

    def __pre_environment(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = core.Model().to(self.device)
        self.train_loader, self.vali_loader, self.test_loader = core.get_data_loader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            normalized=self.normalized,
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def __fit(self) -> None:
        """
        訓練model
        """
        scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.25, patience=2
        )
        min = float("inf")
        for _ in range(self.epochs):
            self.__train()
            loss = self.__validate()
            scheduler.step(loss)
            print(f"loss: {loss:.4f}")
            if loss < min and self.save_flag:
                torch.save(self.model.state_dict(), self.save_path)

    def __train(self) -> None:
        self.model.train()
        with tqdm(
            self.train_loader, ncols=100, leave=False, desc="Train", unit="batchs"
        ) as pbar:
            for labels, features in pbar:
                features, labels = (
                    features.to(self.device),
                    labels.to(self.device),
                )
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()

    def fit(self) -> None:
        """
        訓練model
        """
        try:
            self.__fit()
        except torch.cuda.OutOfMemoryError:
            print("Error OutOfMemoryError")
            self.batch_size = self.batch_size // 2
            if self.batch_size == 0:
                print("Error batch size is 0")
                exit(1)
            print(f"New batch size: {self.batch_size}")
            self.__pre_environment()
            self.fit()

    def __validate(self) -> float:
        self.model.eval()
        test_loss = 0
        with (
            torch.no_grad(),
            tqdm(
                self.vali_loader, unit="batchs", ncols=100, leave=False, desc="Validate"
            ) as pbar,
        ):
            for labels, features in pbar:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), labels)
                test_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        return test_loss / len(self.vali_loader)

    def test(self) -> None:
        """
        測試model
        """
        print(f"data mean: {self.mean}, data std: {self.std}")
        diff = []
        predict = []
        row_data = []
        self.model.to(self.device)
        test_loss = 0
        self.model.eval()
        with torch.no_grad():
            for labels, features in track(
                self.test_loader, description="[cyan]Testing..."
            ):
                row_data.append(self.mean * labels[0] + self.std)
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                predict.append(
                    self.mean * outputs[0].to(torch.device("cpu")) + self.std
                )
                loss = self.criterion(outputs.squeeze(), labels)
                test_loss += loss
        for row, pre in zip(row_data, predict):
            diff.append(abs(row - pre))
        print(f"Test: \n\tavg loss: {test_loss / len(self.test_loader)}")
        dic = {
            "diff": diff,
            "row_data": row_data,
            "predict": predict,
        }
        df = pl.DataFrame(dic)
        df.write_csv("../reslut.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Hyperparameters")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--normalized", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="model")
    parser.add_argument("--info_path", type=str, default="info")
    args = parser.parse_args()
    for key, val in zip(args._get_kwargs(), args._get_args()):
        print(f"{key}: {val}")
    trainer = Trainer(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_workers=args.num_workers,
        normalized=args.normalized,
        save_path=args.save_path,
        info_path=args.info_path,
    )
    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    main()
