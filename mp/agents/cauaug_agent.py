import os
import time
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator
from tqdm import tqdm
from mp.eval.inference.predict import softmax
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class CauAugAgent(SegmentationAgent):

    def __init__(self, *args, **kwargs):
        if "metrics" not in kwargs:
            kwargs["metrics"] = ["ScoreDice", "ScoreIoU", "ScoreHausdorff"]
        super().__init__(*args, **kwargs)

        self.magnitude_dataset = []
        self.phase_dataset = []
        self.std_dataset = []
        self.mean_dataset = []

        self.anaaug = False
        self.domaug = False

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
        r"""Train a model through its agent. Performs training epochs,"""
        val_best = config["val_best"]
        self.agent_state_dict["epoch"] = init_epoch
        self.anaaug = config["AnaAug"]
        self.domaug = config["DomAug"]

        print("Training on new dataset")
        self.store_new = config["DomAug"]
        self.best_validation_value = 0.0
        self.best_validation_epoch = 0

        for epoch in range(init_epoch, nr_epochs):
            print("Epoch:", epoch)
            self.agent_state_dict["epoch"] = epoch

            acc = self.perform_training_epoch(loss_f, train_dataloader, config, print_run_loss=True)

            if val_best:
                dice = self.track_validation_metrics(dataset_index, loss_f, eval_datasets, save_path, epoch, acc)
                print("validation dice:", dice)
                if dice > self.best_validation_value:
                    self.best_validation_value = dice
                    self.best_validation_epoch = epoch
                    if save_path is not None:
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

        if self.store_new:
            self.domain_knowledge_archive(
                train_dataloader=train_dataloader,
                config=config,
                dataset_index=dataset_index,
            )
            self.store_new = False

        self.track_metrics(nr_epochs, results, loss_f, eval_datasets)
        print("Training finished.")
        self.model.finish()

    def perform_training_epoch(self, loss_f, train_dataloader, config, print_run_loss=False):

        acc = Accumulator("loss")
        start_time = time.time()

        for data in tqdm(train_dataloader, disable=True):
            # Get data
            inputs, targets = self.get_inputs_targets(data)

            # learning new knowldge
            if config["AnaAug"]:
                inputs_new = self.cross_kernel_network(inputs=inputs)
            else:
                inputs_new = inputs
            outputs_new = self.get_outputs_new(inputs_new)
            loss_new = loss_f(outputs_new, targets)

            # learning old knowledge
            if self.model.unet_old != None:
                # distilling semantic information
                outputs_kd = self.get_outputs_old(inputs)
                loss_kd = self.multi_class_cross_entropy_no_softmax(outputs_new, outputs_kd)
                loss_old = loss_kd
                if config["DomAug"]:
                    # self.save_augmentation_sample(targets, "targets.jpg")
                    inputs_old = self.fourier_transformer_generator(inputs)
                    outputs_old = self.get_outputs_new(inputs_old)
                    targets_old = self.get_outputs_old(inputs_old)
                    loss_dom = self.multi_class_cross_entropy_no_softmax(outputs_old, targets_old)
                    loss_old = loss_kd + loss_dom

            else:
                if loss_new.is_cuda:
                    loss_old = torch.zeros(1).to(loss_new.get_device())
                else:
                    loss_old = torch.zeros(1)

            loss = loss_new + config["lambda_d"] * loss_old

            # Optimization step
            self.model.unet_optim.zero_grad()
            loss.backward()
            self.model.unet_optim.step()
            acc.add("loss", float(loss.detach().cpu()), count=len(inputs))
            acc.add("loss_new", float(loss_new.detach().cpu()), count=len(inputs))
            acc.add("loss_old", float(loss_old.detach().cpu()), count=len(inputs))

        if print_run_loss:
            print("running loss: {} - time/epoch {}".format(acc.mean("loss"), round(time.time() - start_time, 4)))
            print("new knowledge loss: {}".format(acc.mean("loss_new")))
            print("old knowledge loss: {}".format(acc.mean("loss_old")))

        # add random on dataloader
        for imgs in train_dataloader:
            break
        for imgs in train_dataloader:
            break

        return acc

    def multi_class_cross_entropy_no_softmax(self, prediction, target):
        r"""Stable Multiclass Cross Entropy with Softmax"""
        return (-(target * torch.log(prediction)).sum(dim=-1)).mean()

    def get_outputs_old(self, inputs):
        r"""Applies a softmax transformation to the model outputs."""
        outputs = self.model.forward_old(inputs)
        outputs = softmax(outputs).clamp(min=1e-08, max=1.0 - 1e-08)
        return outputs

    def get_outputs_new(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model(inputs)
        outputs = softmax(outputs).clamp(min=1e-08, max=1.0 - 1e-08)
        return outputs

    def get_outputs(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        if self.anaaug:
            inputs = self.cross_kernel_network(inputs)
        outputs = self.model(inputs)
        outputs = softmax(outputs).clamp(min=1e-08, max=1.0 - 1e-08)
        return outputs

    def cross_kernel_network(self, inputs):
        r"""Cross kernel network"""
        # shadow layers
        nb, nc, nx, ny = inputs.shape

        w_1 = torch.rand(32, nc, 3, 3).to(inputs.get_device())  # random uniform kernel
        b_1 = torch.rand(32).to(inputs.get_device())  # kernel size is 3
        x_1 = F.conv2d(inputs, w_1, b_1, stride=1, padding=1)
        x_1 = F.leaky_relu(x_1)

        w_2 = torch.rand(32, 32, 1, 1).to(inputs.get_device())  # kernel size is 1
        b_2 = torch.rand(32).to(inputs.get_device())
        x_2 = F.conv2d(x_1, w_2, b_2, stride=1, padding=0)
        x_2 = F.leaky_relu(x_2)

        w_3 = torch.rand(32, 32, 3, 3).to(inputs.get_device())  # spatial integration
        b_3 = torch.rand(32).to(inputs.get_device())
        x_3 = F.conv2d(x_2, w_3, b_3, stride=1, padding=1)
        x_3 = F.leaky_relu(x_3)

        w_4 = torch.rand(nc, 32, 1, 1).to(inputs.get_device())  # channel integration
        b_4 = torch.rand(nc).to(inputs.get_device())
        x_4 = F.conv2d(x_3, w_4, b_4, stride=1, padding=0)
        x_4 = F.leaky_relu(x_4)

        x_4 = (x_4 - torch.min(x_4)) / (torch.max(x_4) - torch.min(x_4))

        # self.save_augmentation_sample(inputs, "anaaug_input.jpg")
        # self.save_augmentation_sample(x_4, "anaaug_output.jpg")

        return x_4

    def fourier_transformer_generator(self, inputs):
        r"""Fourier transformer generator"""
        fft = torch.fft.fft2(inputs)  # current phase

        index = torch.randint(len(self.magnitude_dataset), [])  # uniform sampling
        std_old = self.std_dataset[index].view(1, -1, 1, 1) * 10.0
        mean_old = self.mean_dataset[index].view(1, -1, 1, 1) * 10.0
        # phase_old = self.phase_dataset[index]
        magnitude_old = self.magnitude_dataset[index]  # old magnitude

        new_fft = magnitude_old * torch.exp(1j * torch.angle(fft))
        new_inputs = torch.fft.ifft2(new_fft).real  # inverse fourier transform

        cur_std = torch.std(new_inputs, dim=(2, 3), keepdim=True)
        cur_mean = torch.mean(new_inputs, dim=(2, 3), keepdim=True)
        # match the old domain's statisticel properties
        new_inputs = (new_inputs - cur_mean) / cur_std * std_old + mean_old

        # self.save_augmentation_sample(inputs, "domaug_input.jpg")
        # self.save_augmentation_sample(new_inputs, "domaug_output.jpg")

        return new_inputs

    def domain_knowledge_archive(
        self,
        train_dataloader,
        config,
        dataset_index="",
        out_path="domain_knowledge_archive",
    ):
        r"""Store the domain knowledge of the training set"""

        magnitude_instance = torch.zeros(
            [
                config["batch_size"],
                config["input_dim_c"],
                config["input_dim_hw"],
                config["input_dim_hw"],
            ]
        )
        phase_instance = torch.zeros_like(magnitude_instance)

        std_instance = 0
        mean_instance = 0

        count = 0
        for data in tqdm(train_dataloader, disable=True):
            # Get data
            inputs, _ = self.get_inputs_targets(data)

            std = torch.std(inputs, dim=(2, 3), keepdim=True)
            mean = torch.mean(inputs, dim=(2, 3), keepdim=True)
            std = torch.mean(std, dim=0)
            mean = torch.mean(mean, dim=0)

            fft = torch.fft.fft2(inputs)
            amplitude = torch.abs(fft)
            phase = torch.angle(fft)

            amplitude = torch.mean(amplitude, dim=0)
            phase = torch.mean(phase, dim=0)

            if count == 0:
                magnitude_instance = amplitude
                phase_instance = phase
                std_instance = std
                mean_instance = mean
            else:
                magnitude_instance = magnitude_instance * (count / (count + 1)) + amplitude / (count + 1)
                phase_instance = phase_instance * (count / (count + 1)) + phase / (count + 1)
                std_instance = std_instance * (count / (count + 1)) + std / (count + 1)
                mean_instance = mean_instance * (count / (count + 1)) + mean / (count + 1)
            count = count + 1

        self.magnitude_dataset.append(magnitude_instance)
        self.phase_dataset.append(phase_instance)
        self.std_dataset.append(std_instance)
        self.mean_dataset.append(mean_instance)

        if out_path is not None:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # save to 2D image
            magnitude_jpg = torch.fft.fftshift(self.magnitude_dataset[-1][0].detach().cpu()).numpy()
            phase_jpg = torch.fft.fftshift(self.phase_dataset[-1][0].detach().cpu()).numpy()

            plt.figure()
            plt.imshow(magnitude_jpg, cmap="gray")
            plt.colorbar()
            plt.savefig(
                os.path.join(
                    out_path,
                    config["dataset"] + str(dataset_index) + "_magnitude_domain.jpg",
                )
            )
            plt.close()

            plt.figure()
            plt.imshow(phase_jpg, cmap="gray")
            plt.colorbar()
            plt.savefig(
                os.path.join(
                    out_path,
                    config["dataset"] + str(dataset_index) + "_phase_domain.jpg",
                )
            )
            plt.close()

            # save to 3D image
            x = np.linspace(0, magnitude_jpg.shape[1] - 1, magnitude_jpg.shape[1])
            y = np.linspace(0, magnitude_jpg.shape[0] - 1, magnitude_jpg.shape[0])
            x, y = np.meshgrid(x, y)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            surf = ax.plot_surface(x, y, np.log(magnitude_jpg + 1), cmap="viridis")

            # ax.set_title("Magnitude")
            ax.set_xlabel("Fourier Frequency x")
            ax.set_ylabel("Fourier Frequency y")
            # ax.set_zlabel("Magnitude (Log scale)")

            # fig.colorbar(surf, shrink=0.5, aspect=5, label="Magnitude (Log scale)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    out_path,
                    config["dataset"] + str(dataset_index) + "_magnitude_3d.jpg",
                )
            )
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(x, y, phase_jpg, cmap="viridis")
            # ax.set_title("Phase")
            ax.set_xlabel("Fourier Frequency x")
            ax.set_ylabel("Fourier Frequency y")
            ax.set_zlabel("Phase")
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, config["dataset"] + str(dataset_index) + "_phase_3d.jpg"))
            plt.close()

            # save to txt file
            with open(os.path.join(out_path, config["dataset"] + "_mean_std.txt"), "a+") as f:
                f.writelines(str(dataset_index) + ", " + str(mean_instance) + ", " + str(std_instance) + "\n")

    def save_augmentation_sample(self, input_tensor, out_path):
        plt.figure()
        plt.imshow(input_tensor[0, 0].detach().cpu().numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.show()
