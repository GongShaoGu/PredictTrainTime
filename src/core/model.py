import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size: int = 9, output_size: int = 1):
        """
        Args:
            hidden_layer_size: 全連接層的數量
            input_size: 特徵的數量
            output_size: 輸出的維度
        """
        super(Model, self).__init__()
        pass

    def forward(self, x):
        pass


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    from dataloader import get_data_loader

    # _, test_loader = get_data_loader(sequence_length=5, num_workers=8)
    _, _, test_loader = get_data_loader(num_workers=8)
    labels, features = next(iter(test_loader))
    input_size = features.shape[-1]
    model = Model()
    model.to(torch.device("cuda"))
    summary(model, features.shape)
