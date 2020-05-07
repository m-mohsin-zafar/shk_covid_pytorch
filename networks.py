import torch
from torch.nn import AvgPool2d, Conv2d, Linear, Sequential, MaxPool2d, \
    ReLU, Module, BatchNorm2d, Dropout, LocalResponseNorm


class CustomVGG16(Module):
    def __init__(self, num_classes=2):
        super(CustomVGG16, self).__init__()
        self.name = 'CustomVGG16'
        self.num_classes = num_classes
        self.conv_block_1 = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_3 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_4 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_1 = Linear(4 * 4 * 512, 4096)
        self.dropout_1 = Sequential(
            ReLU(inplace=True),
            Dropout(0.5)
        )
        self.fc_2 = Linear(4096, 64)
        self.dropout_2 = Sequential(
            ReLU(inplace=True),
            Dropout(0.5)
        )
        self.fc_3 = Linear(64, self.num_classes)

    def forward(self, x):
        # Shape = (Batch_size, 3, 128, 128)
        out = self.conv_block_1(x)
        # Shape = (Batch_size, 64, 64, 64)
        out = self.conv_block_2(out)
        # Shape = (Batch_size, 128, 32, 32)
        out = self.conv_block_3(out)
        # Shape = (Batch_size, 256, 16, 16)
        out = self.conv_block_4(out)
        # Shape = (Batch_size, 512, 4, 4)
        out = out.reshape(out.size(0), -1)
        # Shape = (Batch_size, 8192 = 512*5*5 )
        out = self.fc_1(out)
        out = self.dropout_1(out)
        out = self.fc_2(out)
        out = self.dropout_2(out)
        out = self.fc_3(out)

        return out


class CovidRENet(Module):
    def __init__(self, num_classes=2):
        super(CovidRENet, self).__init__()
        self.name = 'CovidRENet'
        self.num_classes = num_classes
        self.encoder_1 = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            LocalResponseNorm(5),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_2 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            LocalResponseNorm(5),
            AvgPool2d(kernel_size=2, stride=2)
        )
        self.encoder_3 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            LocalResponseNorm(5),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_4 = Sequential(
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout_1_n_relu = Sequential(
            Dropout(0.5),
            ReLU(inplace=True)
        )
        self.fc_1 = Linear(8 * 8 * 256, 4096)
        self.dropout_2_n_relu = Sequential(
            Dropout(0.2),
            ReLU(inplace=True)
        )
        self.fc_2 = Linear(4096, 256)
        self.dropout_3_n_relu = Sequential(
            Dropout(0.2),
            ReLU(inplace=True)
        )
        self.fc_3 = Linear(256, 64)
        self.dropout_4_n_relu = Sequential(
            Dropout(0.2),
            ReLU(inplace=True)
        )
        self.fc_4 = Linear(64, self.num_classes)

    def forward(self, x):
        # Shape = (Batch_size, 3, 128, 128)
        out = self.encoder_1(x)
        # Shape = (Batch_size, 64, 64, 64)
        out = self.encoder_2(out)
        # Shape = (Batch_size, 128, 32, 32)
        out = self.encoder_3(out)
        # Shape = (Batch_size, 256, 16, 16)
        out = self.encoder_4(out)
        # Shape = (Batch_size, 256, 8, 8)
        out = self.dropout_1_n_relu(out)
        out = out.reshape(out.size(0), -1)
        # Shape = (Batch_size, 16384)
        out = self.fc_1(out)
        out = self.dropout_2_n_relu(out)
        out = self.fc_2(out)
        out = self.dropout_3_n_relu(out)
        out = self.fc_3(out)
        out = self.dropout_4_n_relu(out)
        out = self.fc_4(out)
        return out


if __name__ == '__main__':
    inp = torch.randn(size=(2, 3, 128, 128))
    model = CustomVGG16()
    outp = model(inp)
