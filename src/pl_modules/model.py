from typing import Any, Dict, List, Sequence, Tuple, Union

import gc
import sys
import os
import hydra
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch import nn
from tqdm import tqdm
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients, Occlusion, GuidedGradCam, Saliency
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from src.common.utils import iterate_elements_in_batches, render_images

from src.pl_modules import losses, resnets
from src.pl_modules.network_tools import get_network

# from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay


from typing import Optional

import torch


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> tgm.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps



class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)


def dice_loss(
        input: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return DiceLoss()(input, target)



class MyModel(pl.LightningModule):
    def __init__(
            self,
            cfg: DictConfig,
            name,
            num_classes,
            final_nl,
            loss,
            final_nl_dim,
            plot_gradients_val=True,
            self_supervised=False,
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.name = name
        self.self_supervised = self_supervised
        # self.automatic_optimization = False
        self.num_classes = num_classes
        self.loss = getattr(losses, loss)  # Add this to the config
        self.final_nl_dim = final_nl_dim
        self.plot_gradients_val = plot_gradients_val

        if final_nl:
            self.final_nl = getattr(F, final_nl)
        else:
            self.final_nl = lambda x: x

        self.net = get_network(self.name)  # resnets.resunet()  # get_network(self.name)

        metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_channel_accuracy = metric.clone().cuda()
        self.val_channel_accuracy = metric.clone().cuda()
        self.test_channel_accuracy = metric.clone().cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def step(self, x, y_class, y_channels, filename, class_weight=1.) -> Dict[str, torch.Tensor]:
        class_pred, p0 = self(x)
        if p0 is None:
            cl0, cl1, cl2, channel_loss = None, None, None, None
        else:
            # Do multiple scales to help with mismatches
            c10 = [nn.CrossEntropyLoss(reduction="none")(p0[:, idx], y_channels.long()[:, idx]).mean() for idx in range(y_channels.shape[1])]
            c10 = torch.mean(torch.stack(c10))

            channel_loss = c10  #  + cl0_1 + cl0_2 + cl0_3
            loss = channel_loss  # cl0
        return {
            "channel_logits_0": p0,
            "channel_loss": channel_loss,
            "loss": loss,
            "y_channels": y_channels,
            "x": x,
            "filename": filename}

    def val_step(self, x, y_class, y_channels, filename, class_weight=4.) -> Dict[str, torch.Tensor]:
        class_pred, p0 = self(x)
        if p0 is None:
            loss = class_loss
            cl0, cl1, cl2, channel_loss = None, None, None, None
        else:
            c10 = [nn.CrossEntropyLoss(reduction="none")(p0[:, idx], y_channels.long()[:, idx]).mean() for idx in range(y_channels.shape[1])]
            c10 = torch.mean(torch.stack(c10))

            c10 = c10.detach()
            channel_loss = c10
            loss = c10
            p0 = p0.detach()  # .cpu()

        return {
            "channel_logits_0": p0,
            "loss": loss,
            "y_channels": y_channels,
            "x": x,
            "filename": filename}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y_class, y_channels, filename = batch
        out = self.step(x, y_class, y_channels, filename)
        return out

    def training_step_end(self, out):
        channel_act = out["channel_logits_0"]

        channel_act = channel_act.detach()  # torch.argmax(channel_act, 1)  # self.final_nl(out["channel_logits"], dim=1)

        y_channels = out["y_channels"]
        y_channels_shape = y_channels.shape

        # self.train_channel_accuracy(channel_act.float().cpu(), y_channels.int().cpu())
        mean_loss = out["loss"].mean()  # .item()
        if out["channel_loss"] is not None:
            mean_channel_loss = out["channel_loss"].mean().item()
        else:
            mean_channel_loss = 0.
        self.log_dict(
            {
                "train_loss": mean_loss,
                "train_channel_loss": mean_channel_loss,
            },
            on_step=True,
            on_epoch=False
        )
        return mean_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y_class, y_channels, filename = batch
        with torch.no_grad():
            out = self.val_step(x, y_class, y_channels, filename)
        return out
    
    def validation_step_end(self, out):
        channel_act = out["channel_logits_0"]
        y_channels = out["y_channels"]
        y_channels_shape = y_channels.shape
        mean_loss = out["loss"].mean()
        # self.val_channel_accuracy(channel_act.float(), y_channels.int())
        self.log_dict(
            {
                # "val_channel_acc": self.val_channel_accuracy,
                "val_loss": mean_loss.item(),
            },
        )
        return {
            "image": out["x"],
            "y_channels": out["y_channels"],
            "channel_logits_0": out["channel_logits_0"],
            "val_loss": mean_loss,
        }

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y_class, y_channels, filename = batch
        with torch.no_grad():
            out = self.val_step(x, y_class, y_channels, filename)

        return out

    def test_step_end(self, out):
        channel_act = out["channel_logits"]

        y_channels = out["y_channels"]
        y_channels_shape = y_channels.shape
        # self.test_channel_accuracy(arg_channel_act.float(), y_channels.int())
        mean_loss = out["loss"].mean()
        self.log_dict(
            {
                # "test_channel_acc": self.test_channel_accuracy,
                "test_loss": mean_loss,
            },
        )
        return {
            "image": out["x"],
            "y_channels": out["y_channels"],
            "channel_logits": out["channel_logits"],
            "test_loss": mean_loss,
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:

        self.net.classifier
        integrated_gradients = Saliency(self.net.classifier)
        # integrated_gradients = LayerIntegratedGradients(self, self.net.classifier)
        images, images_feat_viz = [], []
        batch_size = self.cfg.data.datamodule.batch_size.val
        for output_element in iterate_elements_in_batches(
            outputs, batch_size, self.cfg.logging.n_elements_to_log
        ):  
            if output_element["channel_logits_0"] is None:
                return
            output_element["y_channels"] = output_element["y_channels"].detach()  # .cpu()
            if output_element["image"].shape[0] == 2:
                output_element["image"] = torch.concat((output_element["image"], output_element["image"][[1]]), 0)
            # Plot images
            rendered_image = render_images(
                output_element["image"],
                autoshow=False,
                normalize=self.cfg.logging.normalize_visualization)
            caption = f"Input image"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )

            # Plot GT channel 0
            if output_element["y_channels"][[0]].max() <= 1:
                output_element["y_channels"][[0]] = output_element["y_channels"][[0]] * 255.

            rendered_image = render_images(
                output_element["y_channels"][[0]],
                autoshow=False,
                normalize=self.cfg.logging.normalize_visualization)
            caption = f"True output image"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )

            # Plot channel 0
            # plt.imshow(arg_im.cpu().float().numpy().transpose(1, 2, 0) / 255.);plt.show()
            if output_element["channel_logits_0"][[0]].max() <= 1:
                output_element["channel_logits_0"][[0]] = output_element["channel_logits_0"][[0]] * 255.

            if output_element["channel_logits_0"].shape[0] > 1:
                output_element["channel_logits_0"] = output_element["channel_logits_0"][[0]]
            arg_im = torch.argmax(output_element["channel_logits_0"].detach(), 1).float()
            if arg_im.shape[0] == 2:
                arg_im = torch.concat((arg_im, arg_im[[1]]), 0)
            rendered_image = render_images(
                arg_im,
                autoshow=False)  # ,
                # normalize=self.cfg.logging.normalize_visualization)
            caption = f"Pred output image"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )

        self.logger.experiment.log(
            {
                "Validation Images": images,
            },
            step=self.global_step)

    def test_epoch_end(self, outputs: List[Any]) -> None:

        # Now do visualizations
        batch_size = self.cfg.data.datamodule.batch_size.test
        images = []
        images_feat_viz = []
        saved_images = []
        viz_maps = []
        gt = []
        pred = []
        integrated_gradients = GuidedGradCam(self, self.net.layer2[0].conv1)
        # (Pdb) GuidedGradCam(self, self.net.layer4)

        # outputs = [x.pop("filename") for x in outputs]
        # for output_element in tqdm(iterate_elements_in_batches(
        #     outputs, batch_size, self.cfg.logging.n_elements_to_log
        # ), desc="Running visualizations", total=len(outputs)):
        count = 0
        for output in outputs:

            ims = output["image"]
            ys = output["y_true"]
            logits = output["logits"]
            data = []
            for im, y, logit in zip(ims, ys, logits):
                data.append({"image": im, "y_true": y, "logits": logit})

            for output_element in tqdm(data, desc="Running visualizations", total=len(outputs)):
                if count > self.cfg.logging.n_elements_to_log:
                    break
                attributions_ig_nt = integrated_gradients.attribute(
                    output_element["image"].unsqueeze(0),
                    target=output_element["y_true"])
                # attributions_ig_nt = noise_tunnel.attribute(
                #     output_element["image"].unsqueeze(0),
                #     nt_samples=50,
                #     nt_type='smoothgrad_sq',
                #     target=0)  # ,  # output_element["y_true"],
                #     # internal_batch_size=50)
                plottable_attr_dead = np.transpose(attributions_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                # attributions_ig_nt = noise_tunnel.attribute(
                #     output_element["image"].unsqueeze(0),
                #     nt_samples=50,
                #     nt_type='smoothgrad_sq',
                #     target=1)  # ,  # output_element["y_true"],
                #     # internal_batch_size=50)
                # attributions_ig_nt = integrated_gradients.attribute(
                #     output_element["image"].unsqueeze(0),
                #     target=1,  # output_element["y_true"],
                #     strides=(1, 2, 2),
                #     sliding_window_shapes=(1, 2, 2))
                # plottable_attr_live = np.transpose(attributions_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                viz_maps.append((plottable_attr_dead, plottable_attr_dead))  # plottable_attr_live))
                gt.append(output_element["y_true"].cpu().detach().numpy())
                pred.append(output_element["logits"].cpu().detach().numpy())
                # rendered_image = render_images(output_element["image"], autoshow=False)
                saved_images.append(output_element["image"].cpu().detach().numpy())
                count += 1

        # Make directories then save data
        np.savez("attributions", viz=viz_maps, images=saved_images, gt=gt, pred=pred)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        if hasattr(self.cfg.optim.optimizer, "exclude_bn_bias") and \
                self.cfg.optim.optimizer.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.cfg.optim.optimizer.weight_decay)
        else:
            params = self.parameters()

        opt = hydra.utils.instantiate(
            self.cfg.optim.optimizer, params=params, weight_decay=self.cfg.optim.optimizer.weight_decay
        )
        
        if not self.cfg.optim.use_lr_scheduler:
            return opt

        # Handle schedulers if requested
        if 0:  # torch.optim.lr_scheduler.warmup_steps:
            # Right now this is specific to SimCLR
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    opt,
                    linear_warmup_decay(
                        self.cfg.optim.lr_scheduler.warmup_steps,
                        self.cfg.optim.lr_scheduler.total_steps,
                        cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            lr_scheduler = self.cfg.optim.lr_scheduler
        scheduler = hydra.utils.instantiate(lr_scheduler, optimizer=opt)
        return [opt], [scheduler]

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]
