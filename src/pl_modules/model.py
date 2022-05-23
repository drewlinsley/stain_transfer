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

from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay


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

        metric = torchmetrics.Accuracy()
        self.train_channel_accuracy = metric.clone().cuda()
        self.val_channel_accuracy = metric.clone().cuda()
        self.test_channel_accuracy = metric.clone().cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def step(self, x, y_class, y_channels, filename, class_weight=1.) -> Dict[str, torch.Tensor]:
        if self.self_supervised:
            raise NotImplementedError
            z1, z2 = self.shared_step(x)
            loss = self.loss(z1, z2)
        else:
            class_pred, p0, p1, p2 = self(x)
            # class_pred = class_pred.squeeze(-1).squeeze(-1)
            # print(class_pred.shape, channel_preds.shape, y_class.shape, y_channels.shape)
            # if len(y_class) == 1:
            #     # Weird bug and fix
            #     class_loss = self.loss(class_pred.repeat(2, 1), y_class.repeat(2))
            # else:
            #     class_loss = self.loss(class_pred, y_class)
            if p0 is None:
                cl0, cl1, cl2, channel_loss = None, None, None, None
            else:
                cl0 = nn.CrossEntropyLoss(reduction="none")(p0, y_channels[:, 0].long()).mean()
                channel_loss = cl0
                loss = cl0
        return {
            "channel_logits_0": p0,
            "channel_loss": channel_loss,
            "loss": loss,
            "y_channels": y_channels,
            "x": x,
            "filename": filename}

    def val_step(self, x, y_class, y_channels, filename, class_weight=4.) -> Dict[str, torch.Tensor]:
        class_pred, p0, p1, p2 = self(x)
        if p0 is None:
            loss = class_loss
            cl0, cl1, cl2, channel_loss = None, None, None, None
        else:
            cl0 = nn.CrossEntropyLoss(reduction="none")(p0, y_channels[:, 0].long()).mean()

            channel_loss = cl0
            loss = cl0
            p0 = p0.detach().cpu()

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
        self.val_channel_accuracy(arg_channel_act.float(), y_channels.int())
        self.log_dict(
            {
                "val_channel_acc": self.val_channel_accuracy,
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
        return
        x, y_class, y_channels, filename = batch
        with torch.no_grad():
            out = self.val_step(x, y_class, y_channels, filename)

        return out

    def test_step_end(self, out):
        channel_act = out["channel_logits"]

        y_channels = out["y_channels"]
        y_channels_shape = y_channels.shape
        self.test_channel_accuracy(arg_channel_act.float(), y_channels.int())
        mean_loss = out["loss"].mean()
        self.log_dict(
            {
                "test_channel_acc": self.test_channel_accuracy,
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
            # Plot images
            rendered_image = render_images(
                output_element["image"],
                autoshow=False,
                normalize=self.cfg.logging.normalize_visualization)
            caption = f"H&E image"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )

            # Plot GT channel 0
            rendered_image = render_images(
                output_element["y_channels"][[0]],
                autoshow=False,
                normalize=self.cfg.logging.normalize_visualization)
            caption = f"True PolyT image"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )

            # Plot channel 0
            rendered_image = render_images(
                torch.argmax(output_element["channel_logits_0"].detach(), 0).float(),
                autoshow=False,
                normalize=self.cfg.logging.normalize_visualization)
            caption = f"Pred PolyT image"
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
        if torch.optim.lr_scheduler.warmup_steps:
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
