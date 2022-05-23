import math
from argparse import ArgumentParser

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torch.nn as nn
from src.pl_modules.modules import ResidualConv, Upsample


def resunet(in_channel=1):
    # assert num_classes is not None, "You must pass the number of classes to your model."
    model = ResUnet(in_channel=in_channel)  # , num_classes=num_classes)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model.maxpool = nn.Identity()
    return model


def resunet_restaining(in_channel=1):
    # assert num_classes is not None, "You must pass the number of classes to your model."
    model = ResUnet_restaining(in_channel=in_channel)  # , num_classes=num_classes)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model.maxpool = nn.Identity()
    return model


def resunet_control(in_channel=1):
    # assert num_classes is not None, "You must pass the number of classes to your model."
    model = ResUnet(in_channel=in_channel, control=True)  # , num_classes=num_classes)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model.maxpool = nn.Identity()
    return model


class ResUnet_restaining(nn.Module):
    def __init__(self, in_channel=1, filters=[64, 128, 256, 384], control=False):
        super(ResUnet_restaining, self).__init__()
        self.control = control

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        # AD classifier
        self.classifier = nn.Linear(filters[3], 2)
        if control:
            return

        # Ch_0
        self.upsample_0_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv_0_1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_0_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv_0_2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_0_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv_0_3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer_0 = nn.Conv2d(filters[0], 2, 1, 1)

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)

        # Classifer
        mean_vec = x4.mean(dim=(2, 3))
        ad_pred = self.classifier(mean_vec)
        if self.control:
            return ad_pred, None, None, None

        # Decode 0
        x4_0 = self.upsample_0_1(x4)
        x5_0 = torch.cat([x4_0, x3], dim=1)
        x6_0 = self.up_residual_conv_0_1(x5_0)
        x6_0 = self.upsample_0_2(x6_0)
        x7_0 = torch.cat([x6_0, x2], dim=1)
        x8_0 = self.up_residual_conv_0_2(x7_0)
        x8_0 = self.upsample_0_3(x8_0)
        x9_0 = torch.cat([x8_0, x1], dim=1)
        x10_0 = self.up_residual_conv_0_3(x9_0)
        output_0 = self.output_layer_0(x10_0)

        # # Combine preds
        # channel_preds = torch.stack((output_0, output_1, output_2), 2)  # Channel cat

        # Return outputs
        return ad_pred, output_0, output_0, output_0  # channel_preds



class ResUnet(nn.Module):
    def __init__(self, in_channel=1, filters=[64, 128, 256, 384], control=False):
        super(ResUnet, self).__init__()
        self.control = control

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        # AD classifier
        self.classifier = nn.Linear(filters[3], 2)
        if control:
            return

        # Ch_0
        self.upsample_0_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv_0_1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_0_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv_0_2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_0_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv_0_3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer_0 = nn.Conv2d(filters[0], 2, 1, 1)

        # Ch_1
        self.upsample_1_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv_1_1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_1_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv_1_2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_1_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv_1_3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer_1 = nn.Conv2d(filters[0], 2, 1, 1)

        # Ch_2
        self.upsample_2_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv_2_1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv_2_2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_2_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv_2_3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer_2 = nn.Conv2d(filters[0], 2, 1, 1)

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)

        # Classifer
        mean_vec = x4.mean(dim=(2, 3))
        ad_pred = self.classifier(mean_vec)
        if self.control:
            return ad_pred, None, None, None

        # Decode 0
        x4_0 = self.upsample_0_1(x4)
        x5_0 = torch.cat([x4_0, x3], dim=1)
        x6_0 = self.up_residual_conv_0_1(x5_0)
        x6_0 = self.upsample_0_2(x6_0)
        x7_0 = torch.cat([x6_0, x2], dim=1)
        x8_0 = self.up_residual_conv_0_2(x7_0)
        x8_0 = self.upsample_0_3(x8_0)
        x9_0 = torch.cat([x8_0, x1], dim=1)
        x10_0 = self.up_residual_conv_0_3(x9_0)
        output_0 = self.output_layer_0(x10_0)

        # Decode 1
        x4_1 = self.upsample_1_1(x4)
        x5_1 = torch.cat([x4_1, x3], dim=1)
        x6_1 = self.up_residual_conv_1_1(x5_1)
        x6_1 = self.upsample_1_2(x6_1)
        x7_1 = torch.cat([x6_1, x2], dim=1)
        x8_1 = self.up_residual_conv_1_2(x7_1)
        x8_1 = self.upsample_1_3(x8_1)
        x9_1 = torch.cat([x8_1, x1], dim=1)
        x10_1 = self.up_residual_conv_1_3(x9_1)
        output_1 = self.output_layer_1(x10_1)

        # Decode 2
        x4_2 = self.upsample_2_1(x4)
        x5_2 = torch.cat([x4_2, x3], dim=1)
        x6_2 = self.up_residual_conv_2_1(x5_2)
        x6_2 = self.upsample_2_2(x6_2)
        x7_2 = torch.cat([x6_2, x2], dim=1)
        x8_2 = self.up_residual_conv_2_2(x7_2)
        x8_2 = self.upsample_2_3(x8_2)
        x9_2 = torch.cat([x8_2, x1], dim=1)
        x10_2 = self.up_residual_conv_2_3(x9_2)
        output_2 = self.output_layer_2(x10_2)

        # # Combine preds
        # channel_preds = torch.stack((output_0, output_1, output_2), 2)  # Channel cat

        # Return outputs
        return ad_pred, output_0, output_1, output_2  # channel_preds
