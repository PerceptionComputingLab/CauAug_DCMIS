import os
import time
from tqdm import tqdm
import torch

from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax

from mp.utils.dataaug import AdvBias  # interpolating random control points, implemented by Chen Chen et al.
from mp.utils.dataaug import rescale_intensity
from mp.utils.dataaug import GINGroupConv  # gin

import torch.nn.functional as F


class GINIPAAgent(SegmentationAgent):
    r"""Extension of Segmentation Agent to support Knowledge Distillation
    as porposed in Incremental learning techniques for semantic segmentation by Michieli, U., Zanuttigh, P., 2019
    """

    def __init__(self, *args, **kwargs):
        if "metrics" not in kwargs:
            kwargs["metrics"] = ["ScoreDice", "ScoreIoU", "ScoreHausdorff"]
        super().__init__(*args, **kwargs)

        blender_cofig = {
            "epsilon": 0.3,
            "xi": 1e-6,
            "control_point_spacing": [24, 24],  # 24*2=48, 1/4 of image size
            "downscale": 2,
            "data_size": [4, 1, 192, 192],
            "interpolation_order": 2,
            "init_mode": "gaussian",
            "space": "log",
        }
        self.img_transform_node = GINGroupConv(
            out_channel=1,
            n_layer=4,
            interm_channel=2,
            out_norm="frob",
            device=self.device,
        ).to(self.device)
        self.blender_node = AdvBias(blender_cofig, use_gpu=self.device)  # IPA
        self.blender_node.init_parameters()
        self.criterionCons = torch.nn.KLDivLoss()
        self._nb_current = 4

    def train(
        self,
        results,
        loss_f,
        train_dataloader,
        test_dataloader,
        config,
        init_epoch=0,
        nr_epochs=100,
        eval_datasets=dict(),
        save_path="",
        dataset_index=0,
        exp_path="",
    ):
        r"""Train a model through its agent. Performs training epochs,
        tracks metrics and saves model states.

        Args:
            results (mp.eval.result.Result): results object to track progress
            loss_f (mp.eval.losses.loss_abstract.LossAbstract): loss function for the segmenter
            train_dataloader (torch.utils.data.DataLoader): dataloader of training set
            test_dataloader (torch.utils.data.DataLoader): dataloader of test set
            eval_datasets (torch.utils.data.DataLoader): dataloader of evaluation set
            config (dict): configuration dictionary from parsed arguments
            init_epoch (int): initial epoch
            nr_epochs (int): number of epochs to train for
            save_path (str): save path for saving model, etc.
        """

        run_loss_print_interval = config["run_loss_print_interval"]
        save_interval = config["save_interval"]
        display_interval = config["display_interval"]
        # device_ids = config['device_ids']
        val_best = config["val_best"]
        self.agent_state_dict["epoch"] = init_epoch

        self.best_validation_value = 0.0
        self.best_validation_epoch = 0

        for epoch in range(init_epoch, nr_epochs):
            print("Epoch:", epoch)
            self.agent_state_dict["epoch"] = epoch

            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            acc = self.perform_training_epoch(loss_f, train_dataloader, config, epoch, print_run_loss=print_run_loss)
            if val_best:
                dice = self.track_validation_metrics(dataset_index, loss_f, eval_datasets, save_path, epoch, acc)
                print("validation dice:", dice)
                if dice > self.best_validation_value:
                    self.best_validation_value = dice
                    self.best_validation_epoch = epoch
                    self.save_state(save_path, epoch + 1)
            else:
                # Save agent and optimizer state
                if (epoch + 1) % save_interval == 0 and save_path is not None:
                    self.save_state(save_path, epoch + 1)
            self.model.unet_scheduler.step()

        if val_best:
            self.restore_state(exp_path, self.best_validation_epoch + 1)
            with open(os.path.join(exp_path, "val_track.txt"), "a+") as f:
                f.writelines(str(self.best_validation_epoch + 1) + "\n")
            print(
                "best epoch is ",
                self.best_validation_epoch + 1,
                "; best val dice is",
                self.best_validation_value,
            )

        self.track_metrics(nr_epochs, results, loss_f, eval_datasets)

        self.model.finish()

    def perform_training_epoch(self, loss_f, train_dataloader, config, epoch, print_run_loss=False):
        r"""Perform a training epoch

        Args:
            loss_f (mp.eval.losses.loss_abstract.LossAbstract): loss function for the segmenter
            train_dataloader (torch.utils.data.DataLoader): dataloader of training set
            config (dict): configuration dictionary from parsed arguments
            print_run_loss (boolean): whether to print running loss

        Returns:
            acc (mp.eval.accumulator.Accumulator): accumulator holding losses
        """
        acc = Accumulator("loss")
        start_time = time.time()

        for data in tqdm(train_dataloader, disable=True):
            # Get data
            inputs, targets = self.get_inputs_targets(data)

            # random no-linear augmentation
            self._nb_current = inputs.shape[0]  # batch size of the current batch
            if len(inputs) < config["batch_size"]:
                continue

            # gin
            input_buffer = torch.cat([self.img_transform_node(inputs) for ii in range(3)], dim=0)

            self.blender_node.init_parameters()
            blend_mask = rescale_intensity(self.blender_node.bias_field).repeat(1, 1, 1, 1)

            # spatially-variable blending
            input_cp1 = input_buffer[: self._nb_current].clone().detach() * blend_mask + input_buffer[
                self._nb_current : self._nb_current * 2
            ].clone().detach() * (1.0 - blend_mask)
            input_cp2 = (
                input_buffer[: self._nb_current] * (1 - blend_mask)
                + input_buffer[self._nb_current : self._nb_current * 2] * blend_mask
            )

            input_buffer[: self._nb_current] = input_cp1
            input_buffer[self._nb_current : self._nb_current * 2] = input_cp2

            # import matplotlib.pyplot as plt

            # plt.figure()
            # # input and input_buffer
            # plt.subplot(1, 2, 1)
            # plt.imshow(inputs[0].permute(1, 2, 0).detach().cpu().numpy())
            # plt.subplot(1, 2, 2)
            # plt.imshow(input_buffer[0].permute(1, 2, 0).detach().cpu().numpy())
            # plt.savefig("test.png")
            # exit()

            inputs = input_buffer

            # Forward pass
            pred_all = self.model(inputs)
            outputs = softmax(pred_all[: self._nb_current]).clamp(min=1e-08, max=1.0 - 1e-08)

            loss_consist = self.forward_consistency(pred_all)

            # Optimization step
            self.model.unet_optim.zero_grad()

            loss_seg = loss_f(outputs, targets) + loss_consist

            if self.model.unet_old != None:
                outputs_old = self.get_outputs_simple(inputs[: self._nb_current])
                loss_distill = self.multi_class_cross_entropy_no_softmax(outputs, outputs_old)
            else:
                if loss_seg.is_cuda:
                    loss_distill = torch.zeros(1).to(loss_seg.get_device())
                else:
                    loss_distill = torch.zeros(1)

            loss = loss_seg + config["lambda_d"] * loss_distill
            loss.backward()

            self.model.unet_optim.step()

            acc.add("loss", float(loss.detach().cpu()), count=len(inputs))
            acc.add("loss_seg", float(loss_seg.detach().cpu()), count=len(inputs))
            acc.add("loss_distill", float(loss_distill.detach().cpu()), count=len(inputs))

        # self.model.unet_scheduler.step()

        if print_run_loss:
            print(
                "\nrunning loss: {} - distill {} - time/epoch {}".format(
                    acc.mean("loss"),
                    acc.mean("loss_distill"),
                    round(time.time() - start_time, 4),
                )
            )

        return acc

    def multi_class_cross_entropy_no_softmax(self, prediction, target):
        r"""Stable Multiclass Cross Entropy with Softmax

        Args:
            prediction (torch.Tensor): network outputs w/ softmax
            target (torch.Tensor): label OHE

        Returns:
            (torch.Tensor) computed loss
        """
        return (-(target * torch.log(prediction)).sum(dim=-1)).mean()

    def get_outputs_simple(self, inputs):
        r"""Applies a softmax transformation to the model outputs."""
        outputs = self.model.forward_old(inputs)
        outputs = softmax(outputs)
        return outputs

    def forward_consistency(self, pred_all):
        """
        KL-term, enforcing conditional distribution remains unchanged regardless of interventions applied
        """
        lambda_consist = 10.0

        pred_all_prob = F.softmax(pred_all, dim=1)
        pred_avg = (
            1.0
            / 3
            * (
                pred_all_prob[: self._nb_current]
                + pred_all_prob[self._nb_current : self._nb_current * 2]
                + pred_all_prob[self._nb_current * 2 :]
            )
        )  # efficient implementation inspired by Xu et al. (Randconv)
        pred_avg = torch.cat([pred_avg for ii in range(3)], dim=0)
        pred_all = F.log_softmax(
            pred_all, dim=1
        )  # according to pytorch 1.3 documentation, input is log_prob, target is prob
        loss_consist = self.criterionCons(pred_all, pred_avg)
        return lambda_consist * loss_consist
