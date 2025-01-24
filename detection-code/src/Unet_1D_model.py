import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=padding)
        # self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=padding)
        # self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        conv_kernel_size=3,
        padding=1,
        scaling_kernel_size=2,
        dropout=0.25,
    ):
        super().__init__()
        self.conv = conv_block(in_c, out_c, conv_kernel_size, padding)
        self.pool = nn.MaxPool1d((scaling_kernel_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        p = self.dropout(p)
        return x, p


class decoder_block(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        conv_kernel_size=3,
        padding=1,
        scaling_kernel_size=2,
        dropout=0.25,
    ):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_c, out_c, kernel_size=scaling_kernel_size, stride=2, padding=0
        )
        self.conv = conv_block(
            out_c + out_c, out_c, kernel_size=conv_kernel_size, padding=padding
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class build_unet(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        layer_n=32,
        conv_kernel_size=3,
        scaling_kernel_size=2,
        dropout=0.25,
    ):
        super().__init__()
        self.layer_n = layer_n
        self.conv_kernel_size = conv_kernel_size
        self.scaling_kernel_size = scaling_kernel_size
        """ Encoder """
        self.e1 = encoder_block(
            input_channels,
            layer_n,
            conv_kernel_size,
            1,
            scaling_kernel_size,
            dropout=dropout,
        )
        self.e2 = encoder_block(
            layer_n,
            int(layer_n * 2),
            conv_kernel_size,
            1,
            scaling_kernel_size,
            dropout=dropout,
        )
        self.e3 = encoder_block(
            int(layer_n * 2),
            int(layer_n * 3),
            conv_kernel_size,
            1,
            scaling_kernel_size,
            dropout=dropout,
        )
        self.e4 = encoder_block(
            int(layer_n * 3),
            int(layer_n * 4),
            conv_kernel_size,
            1,
            scaling_kernel_size,
            dropout=dropout,
        )
        """ Bottleneck """
        self.b = conv_block(int(layer_n * 4), int(layer_n * 5), conv_kernel_size, 1)
        """ Decoder """
        self.d1 = decoder_block(
            int(layer_n * 5),
            int(layer_n * 4),
            conv_kernel_size,
            1,
            scaling_kernel_size,
            dropout=dropout,
        )
        self.d2 = decoder_block(
            int(layer_n * 4),
            int(layer_n * 3),
            conv_kernel_size,
            1,
            scaling_kernel_size,
            dropout=dropout,
        )
        self.d3 = decoder_block(
            int(layer_n * 3),
            int(layer_n * 2),
            conv_kernel_size,
            1,
            scaling_kernel_size,
            dropout=dropout,
        )
        self.d4 = decoder_block(
            int(layer_n * 2),
            layer_n,
            conv_kernel_size,
            1,
            scaling_kernel_size,
            dropout=dropout,
        )
        """ Classifier """
        self.outputs = nn.Conv1d(layer_n, num_classes, kernel_size=1, padding=0)

    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs
