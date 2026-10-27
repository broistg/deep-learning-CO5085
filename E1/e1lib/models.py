"""E1 models (Ly's notebook code, moved here): softmax classifier, MLP 3072-64-32-10, MiniVGG.
Only change: the layers no longer take a global `device=device` (the caller does `.to(device)`)."""
import torch.nn as nn


class SoftmaxClassifier(nn.Module):
  def __init__(self, input_dim , num_labels):
    super().__init__()

    # 3 channels x 32 x 32 = 3072 features
    self.model=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=input_dim,
                  out_features=num_labels)
    )

  def forward(self, x):
    return self.model(x)


class MLP(nn.Module):
  def __init__(self, input_dim, hidden_size=128, num_labels=10):
    super().__init__()

    # CIFAR-10 input: 3 * 32 * 32 = 3072
    self.model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=input_dim,
                  out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size,
                  out_features=hidden_size//2),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size//2,
                  out_features=num_labels)

    )

  def forward(self, x):
    return self.model(x)


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3):
    super().__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(in_channels=out_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

  def forward(self, x):
    return self.conv(x)

class MiniVGG(nn.Module):
  def __init__(self, input_dim, hidden_units=32, num_classes=10, kernel_size=3):
    super().__init__()

    self.features=nn.Sequential(
        ConvBlock(in_channels=input_dim,
                  out_channels=hidden_units,
                  kernel_size=kernel_size),
        ConvBlock(in_channels=hidden_units,
                  out_channels=hidden_units*2))
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*2*8*8,
                  out_features=num_classes)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x


# name -> constructor, with the sizes used in the original notebook
MODELS = {
    "softmax": lambda: SoftmaxClassifier(input_dim=32 * 32 * 3, num_labels=10),
    "mlp": lambda: MLP(input_dim=32 * 32 * 3, hidden_size=64, num_labels=10),
    "minivgg": lambda: MiniVGG(input_dim=3, hidden_units=16, num_classes=10, kernel_size=3),
}
