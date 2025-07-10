import torch
from torch.nn import functional as F
import torch.fft
import torch.nn as nn
import numpy as np


def rescale_intensity(data, new_min=0, new_max=1, eps=1e-20):
    """
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    """
    bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
    data = data.view(bs * c, -1)
    # pytorch 1.3
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values

    # co1818: in adjust to pytorch 0.4 for wtbenv
    # old_max = torch.max(data, dim=1, keepdim=True)[0]
    # old_min = torch.min(data, dim=1, keepdim=True)[0]

    new_data = (data - old_min + eps) / (old_max - old_min + eps) * (new_max - new_min) + new_min
    new_data = new_data.view(bs, c, h, w)
    return new_data


class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(
        self, out_channel=32, in_channel=3, scale_pool=[1, 3], layer_id=0, use_act=True, requires_grad=False, **kwargs
    ):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_pool = scale_pool
        self.layer_id = layer_id
        self.use_act = use_act
        self.requires_grad = requires_grad
        assert requires_grad == False

    def forward(self, x_in, requires_grad=False):
        """
        Args:
            x_in: [ nb (original), nc (original), nx, ny ]
        """
        # random size of kernel
        idx_k = torch.randint(high=len(self.scale_pool), size=(1,))
        k = self.scale_pool[idx_k[0]]

        nb, nc, nx, ny = x_in.shape

        ker = torch.randn([self.out_channel * nb, self.in_channel, k, k], requires_grad=self.requires_grad).to(
            x_in.device
        )
        shift = torch.randn([self.out_channel * nb, 1, 1], requires_grad=self.requires_grad).to(x_in.device) * 1.0

        x_in = x_in.view(1, nb * nc, nx, ny)
        x_conv = F.conv2d(x_in, ker, stride=1, padding=k // 2, dilation=1, groups=nb)
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny)
        return x_conv


class GINGroupConv(nn.Module):
    def __init__(
        self,
        out_channel=1,
        in_channel=1,
        interm_channel=2,
        scale_pool=[1, 3],
        n_layer=4,
        out_norm="frob",
        device="cuda",
    ):
        """
        GIN
        """
        super(GINGroupConv, self).__init__()
        self.scale_pool = scale_pool  # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel

        self.layers.append(
            GradlessGCReplayNonlinBlock(
                out_channel=interm_channel, in_channel=in_channel, scale_pool=scale_pool, layer_id=0
            ).to(device)
        )
        for ii in range(n_layer - 2):
            self.layers.append(
                GradlessGCReplayNonlinBlock(
                    out_channel=interm_channel, in_channel=interm_channel, scale_pool=scale_pool, layer_id=ii + 1
                ).to(device)
            )
        self.layers.append(
            GradlessGCReplayNonlinBlock(
                out_channel=out_channel,
                in_channel=interm_channel,
                scale_pool=scale_pool,
                layer_id=n_layer - 1,
                use_act=False,
            ).to(device)
        )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim=0)

        nb, nc, nx, ny = x_in.shape

        alphas = torch.rand(nb)[:, None, None, None]  # nb, 1, 1, 1
        alphas = alphas.repeat(1, nc, 1, 1).to(x_in.device)  # nb, nc, 1, 1

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        mixed = alphas * x + (1.0 - alphas) * x_in

        if self.out_norm == "frob":
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p="fro", keepdim=False)
            _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1, -2), p="fro", keepdim=False)
            _self_frob = _self_frob[:, None, None, None].repeat(1, self.out_channel, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed


def bspline_kernel_2d(sigma=[1, 1], order=2, asTensor=False, dtype=torch.float32, device="gpu"):
    """
    generate bspline 2D kernel matrix.
    From wiki: https://en.wikipedia.org/wiki/B-spline, Fast b-spline interpolation on a uniform sample domain can be
    done by iterative mean-filtering
    :param sigma: tuple integers, control smoothness
    :param order: the order of interpolation
    :param asTensor:
    :param dtype: data type
    :param use_gpu: bool
    :return:
    """
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma)

    for i in range(1, order + 1):
        kernel = F.conv2d(kernel, kernel_ones, padding=(i * padding).tolist()) / ((sigma[0] * sigma[1]))

    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()


class AdvTransformBase(torch.nn.Module):
    """
    Adv Transformer base
    """

    def __init__(
        self, config_dict={"size": 1, "mean": 0, "std": 0.1, "xi": 1e-6}, use_gpu: str = "cuda:0", debug: bool = False
    ):
        """ """
        super(AdvTransformBase, self).__init__()
        self.config_dict = config_dict
        self.param = None
        self.is_training = False
        self.use_gpu = use_gpu
        self.debug = debug
        if self.use_gpu:
            self.device = use_gpu
        else:
            self.device = torch.device("cpu")

    def init_config(self, config_dict):
        """
        initialize a set of transformation configuration parameters
        """
        if self.debug:
            print("init base class")
        self.size = config_dict["size"]
        self.mean = config_dict["mean"]
        self.std = config_dict["std"]
        self.xi = config_dict["xi"]

    def init_parameters(self):
        """
        initialize transformation parameters
        return random transformaion parameters
        """
        self.init_config(self.config_dict)
        noise = torch.randn(self.size, device=self.device, dtype=torch.float32) * self.std + self.mean
        self.param = noise
        return noise

    def set_parameters(self, param):
        self.param = param.detach()

    def make_small_parameters(self):
        self.param = self.xi * self.unit_normalize(self.param)

    def get_parameters(self):
        return self.param

    def train(self):
        self.is_training = True
        self.param = torch.nn.Parameter(self.param, requires_grad=True)

    def eval(self):
        self.param.requires_grad = False
        self.is_training = False

    def rescale_parameters(self):
        self.param = self.xi * self.unit_normalize(self.param)
        return self.param

    def optimize_parameters(self, set=False):
        grad = self.param.grad.sign()
        if self.debug:
            print("grad", grad.size())
        if set:
            self.param = grad.detach()
        return self.param

    def forward(self, data):
        """
        forward the data to get transformed data
        :param data: input images x, N4HW
        :return:
        tensor: transformed images
        """
        assert self.param is not None, "init param before transform data"
        transformed_input = data + self.param
        if self.debug:
            print("transformed", transformed_input.size())
        return transformed_input

    def backward(self, data):
        assert self.param is not None, "init param before transform data"
        warped_back_output = data - self.param
        if self.debug:
            print("back:", warped_back_output.size())
        return warped_back_output

    def unit_normalize(self, d, p_type="l2"):
        if p_type == "l1":
            old_size = d.size()
            d_flatten = d.view(d.size(0), -1)
            norm = d_flatten.norm(p=1, dim=1, keepdim=True)
            d_normalized = d_flatten.div(norm.expand_as(d_flatten))
            return d_normalized.view(old_size)
        elif p_type == "infinity":
            d_abs_max = torch.max(torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(d.size(0), 1, 1, 1)
            # print(d_abs_max.size())
            d /= 1e-20 + d_abs_max  ## d' =d/d_max
        if p_type == "l2":
            d_abs_max = torch.max(torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(d.size(0), 1, 1, 1)
            # print(d_abs_max.size())
            d /= 1e-20 + d_abs_max  ## d' =d/d_max
            d /= torch.sqrt(
                1e-6 + torch.sum(torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True)
            )  ##d'/sqrt(d'^2)
        return d


class AdvBias(AdvTransformBase):
    """
    Adv Bias field
    """

    def __init__(
        self,
        config_dict={
            "epsilon": 0.3,
            "xi": 1e-6,
            "control_point_spacing": [32, 32],
            "downscale": 2,
            "data_size": [2, 1, 128, 128],
            "interpolation_order": 3,
            "init_mode": "gaussian",
            "space": "log",
        },
        use_gpu: str = "cuda:0",
        debug: bool = False,
    ):
        super(AdvBias, self).__init__(config_dict=config_dict, use_gpu=use_gpu, debug=debug)
        self.param = None

    def init_config(self, config_dict):
        """
        initialize a set of transformation configuration parameters
        """
        self.epsilon = config_dict["epsilon"]
        self.xi = config_dict["xi"]
        self.data_size = config_dict["data_size"]
        self.control_point_spacing = config_dict["control_point_spacing"]
        self.downscale = config_dict["downscale"]
        self.interpolation_order = config_dict["interpolation_order"]

        self.space = config_dict["space"]
        self.init_mode = config_dict["init_mode"]

        # optional params added by co1818

    def init_parameters(self):
        """
        initialize transformation parameters
        return random transformaion parameters
        """
        self.init_config(self.config_dict)
        self._device = self.use_gpu if self.use_gpu else "cpu"

        self._dim = len(self.control_point_spacing)
        self.spacing = self.control_point_spacing
        self._dtype = torch.float32
        self.batch_size = self.data_size[0]
        self._image_size = np.array([self.data_size[2], self.data_size[3]])
        self.magnitude = self.epsilon
        self.order = self.interpolation_order
        self.downscale = self.downscale  ## reduce image size to save memory

        self.use_log = True  # if self.space == 'log' else False

        ## contruct and initialize control points grid with random values
        self.param, self.interp_kernel, self.bias_field = self.init_bias_field()
        return self.param

    def make_small_parameters(self):
        raise Exception("by co1818: should not call a detaching op")
        self.param = self.unit_normalize(self.param, p_type="l2") * self.xi
        self.param = self.param.detach()
        self.param.requires_grad = True
        return self.param

    def rescale_parameters(self, power_iteration=False):
        ## restrict control points values in the 1-ball space
        self.param = self.unit_normalize(self.param, p_type="l2")

    # def optimize_parameters(self,power_iteration=False, step_size=1): # original implementated by cc215
    # new implementation by co1818, supporting GD

    def optimize_parameters(self, power_iteration=False, step_size=1, upd_direction="GA"):
        """
        Args co1818:
            upd_direction: direction of update. GA for gradient ascend in original implmentation. GD for descent, added by co1818
        """
        if self.debug:
            print("optimize bias")
        grad = self.unit_normalize(self.param.grad, p_type="l2")
        if power_iteration:
            raise Exception("co1818: behavior unknown")
            self.param = grad.clone().detach()
        else:
            if upd_direction == "GA":
                ## Gradient ascent by cc215
                self.param = self.param + step_size * grad.detach()
                self.param = self.param.clone().detach()
            elif upd_direction == "GD":  # added by co1818
                self.param = self.param - step_size * grad.detach()
                self.param = self.param.clone().detach()
            else:
                raise NotImplementedError(f"Unknown updating direction {upd_direction}")

        return self.param

    def set_parameters(self, param):
        raise Exception("by co1818: should not call a detached op")
        self.param = param.detach()

    def forward(self, data):
        """
        forward the data to get transformed data
        :param data: input images x, N4HW
        :return:
        tensor: transformed images
        """
        assert self.param is not None, "init param before transform data"

        bias_field = self.compute_smoothed_bias(self.param)
        bias_field = self.rescale_bias(bias_field, magnitude=self.epsilon)
        self.bias_field = bias_field
        self.diff = bias_field

        ## in case the input image is a multi-channel input.
        if bias_field.size(1) < data.size(1):
            bias_field = bias_field.expand(data.size())

        transformed_input = bias_field * data
        if self.debug:
            print("bias transformed", transformed_input.size())
        return transformed_input

    def backward(self, data):
        if self.debug:
            print("max magnitude", torch.max(torch.abs(self.bias_field - 1)))
        return data

    def predict_forward(self, data):
        return data

    def predict_backward(self, data):
        return data

    def init_bias_field(self, init_mode=None):
        """
                init cp points, interpolation kernel, and resulted bias field.
                :param batch_size:
                :param spacing: tuple of ints
                :param order:
                :return:bias field
                reference:
                bspline interpoplation is adapted from airlab: class _KernelTransformation(_Transformation):
        https://github.com/airlab-unibas/airlab/blob/1a715766e17c812803624d95196092291fa2241d/airlab/transformation/pairwise.py
        """
        if init_mode is None:
            mode = self.init_mode

        ## set up cpoints grid
        self._stride = np.array(self.spacing)  # 32 or 24
        cp_grid = np.ceil(np.divide(self._image_size / (1.0 * self.downscale), self._stride)).astype(
            dtype=int
        )  # co1818: the number of control points
        # new image size after convolution
        inner_image_size = np.multiply(self._stride, cp_grid) - (self._stride - 1)
        # add one control point outside each side, e.g.2 by 2 grid, requires 4 by 4 control points
        cp_grid = cp_grid + 2
        # image size with additional control points
        new_image_size = np.multiply(self._stride, cp_grid) - (self._stride - 1)
        # center image between control points
        image_size_diff = inner_image_size - self._image_size / (1.0 * self.downscale)
        image_size_diff_floor = np.floor((np.abs(image_size_diff) / 2)) * np.sign(image_size_diff)
        self._crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2) * np.sign(image_size_diff)
        self._crop_end = image_size_diff_floor
        self.cp_grid = [self.batch_size, 1] + cp_grid.tolist()

        # initialize control points parameters for optimization
        if mode == "gaussian":
            self.param = torch.ones(*self.cp_grid).normal_(mean=0, std=1)

        # elif mode =='random':
        #     ## diff to the identity
        #     if self.use_log:
        #         raise NotImplementedError
        #     else:
        #         self.param=(torch.rand(*self.cp_grid)*2-1)*self.magnitude

        elif mode == "identity":
            ## static initialization, bias free
            self.param = torch.zeros(*self.cp_grid)
        else:
            raise NotImplementedError

        self.param = self.unit_normalize(self.param, p_type="l2")

        self.param = self.param.to(dtype=self._dtype, device=self._device)

        # convert to integer
        self._stride = self._stride.astype(dtype=int).tolist()
        self._crop_start = self._crop_start.astype(dtype=int)
        self._crop_end = self._crop_end.astype(dtype=int)

        size = [self.batch_size, 1] + new_image_size.astype(dtype=int).tolist()
        ## initialize interpolation kernel
        self.interp_kernel = self.get_bspline_kernel(order=self.order, spacing=self.spacing)
        self.interp_kernel = self.interp_kernel.to(self.param.device)
        self.bias_field = self.compute_smoothed_bias(self.param, padding=self._padding, stride=self._stride)
        self.bias_field = self.rescale_bias(self.bias_field)
        if self.debug:
            print("initialize {} control points".format(str(self.param.size())))

        return self.param, self.interp_kernel, self.bias_field

    # NOTE: this is added by co1818 for convenience
    def reset_bias_value(self):
        """
        Reset param (ct point values)
        """
        del self.param
        del self.bias_field

        mode = self.init_mode
        if mode == "gaussian":
            self.param = torch.ones(*self.cp_grid).normal_(mean=0, std=1)

        elif mode == "identity":
            ## static initialization, bias free
            self.param = torch.zeros(*self.cp_grid)
        else:
            raise NotImplementedError

        self.param = self.unit_normalize(self.param, p_type="l2")
        self.param = self.param.to(dtype=self._dtype, device=self._device)

        self.bias_field = self.compute_smoothed_bias(self.param, padding=self._padding, stride=self._stride)
        self.bias_field = self.rescale_bias(self.bias_field)

        return self.param, self.bias_field

    def compute_smoothed_bias(self, cpoint=None, interpolation_kernel=None, padding=None, stride=None):
        """
        generate bias field given the cppints N*1*k*l
        :return: bias field bs*1*H*W
        """
        if interpolation_kernel is None:
            interpolation_kernel = self.interp_kernel
        if padding is None:
            padding = self._padding
        if stride is None:
            stride = self._stride
        if cpoint is None:
            cpoint = self.param

        bias_field = F.conv_transpose2d(cpoint, interpolation_kernel, padding=padding, stride=stride, groups=1)
        # crop bias
        bias_field_tmp = bias_field[
            :,
            :,
            stride[0] + self._crop_start[0] : -stride[0] - self._crop_end[0],
            stride[1] + self._crop_start[1] : -stride[1] - self._crop_end[1],
        ]

        ## recover bias field to original image resolution for efficiency.
        if self.debug:
            print("after intep, size:", bias_field_tmp.size())
        scale_factor = self._image_size[0] / bias_field_tmp.size(2)
        if scale_factor > 1:
            upsampler = torch.nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
            diff_bias = upsampler(bias_field_tmp)
        else:
            diff_bias = bias_field_tmp

        bias_field = torch.exp(diff_bias)

        return bias_field

    def rescale_bias(self, bias_field, magnitude=None):
        """[summary]
        rescale the bias field so that it values fall in [1-magnitude, 1+magnitude]
        Args:
            bias_field ([torch 4d tensor]): [description]
            magnitude ([scalar], optional): [description]. Defaults to use predefined value.

        Returns:
            [type]: [description]
        """
        if magnitude is None:
            magnitude = self.magnitude
        assert magnitude > 0

        bias_field = rescale_intensity(bias_field, 1 - magnitude, 1 + magnitude)

        if self.debug:
            print("L_infinity: max|bias-id|", torch.max(torch.abs(bias_field - 1)))
        return bias_field

    def get_bspline_kernel(self, spacing, order=3):
        """

        :param order init: bspline order, default to 3
        :param spacing tuple of int: spacing between control points along h and w.
        :return:  kernel matrix
        """
        self._kernel = bspline_kernel_2d(spacing, order=order, asTensor=True, dtype=self._dtype, device=self._device)
        self._padding = (np.array(self._kernel.size()) - 1) / 2
        self._padding = self._padding.astype(dtype=int).tolist()
        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.to(dtype=self._dtype, device=self._device)
        return self._kernel

    def get_name(self):
        return "bias"

    def is_geometric(self):
        return 0


def batch_augment_vectorized(
    batch,
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    gamma_range=(0.8, 1.2),
    noise_std_range=(0.0, 0.1),
):
    """
    对一个 batch 内的图像同时进行几何变换、强度变换以及添加噪声（鲁棒性增强），
    全部通过向量化操作实现，无需 for 循环。

    参数:
      batch: Tensor，形状为 (B, C, H, W)，且像素值在 [0, 1]
      brightness_range: 亮度调整因子范围
      contrast_range: 对比度调整因子范围
      gamma_range: Gamma 调整因子范围
      noise_std_range: 高斯噪声标准差范围

    返回:
      batch_aug: 增强后的图像，Tensor 形状为 (B, C, H, W)
    """
    B, C, H, W = batch.shape
    device = batch.device

    # ----- 1. 强度变换 -----
    # 为每张图像随机生成亮度、对比度、Gamma 调整因子（形状均为 (B, 1, 1, 1)）
    brightness_factors = torch.empty(B, 1, 1, 1, device=device).uniform_(brightness_range[0], brightness_range[1])
    contrast_factors = torch.empty(B, 1, 1, 1, device=device).uniform_(contrast_range[0], contrast_range[1])
    gamma_factors = torch.empty(B, 1, 1, 1, device=device).uniform_(gamma_range[0], gamma_range[1])

    # 调整亮度：直接乘以因子
    batch_intensity = batch * brightness_factors
    # 调整对比度：先计算每张图像的均值，然后进行线性调整
    mean = batch_intensity.mean(dim=[1, 2, 3], keepdim=True)
    batch_intensity = (batch_intensity - mean) * contrast_factors + mean
    # 调整 Gamma：保证输入非负后进行幂运算
    batch_intensity = batch_intensity.clamp(min=0.0)
    batch_intensity = batch_intensity.pow(gamma_factors)

    # ----- 3. 鲁棒性变换（添加噪声） -----
    noise_std = torch.empty(B, 1, 1, 1, device=device).uniform_(noise_std_range[0], noise_std_range[1])
    noise = torch.randn_like(batch_intensity) * noise_std
    batch_aug = batch_intensity + noise
    batch_aug = torch.clamp(batch_aug, 0, 1)

    return batch_aug


import torch
import torch.nn.functional as F


def gaussian_kernel(kernel_size, sigma, device, channels):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    x_grid = coords.repeat(kernel_size, 1)
    y_grid = x_grid.t()
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    return kernel


def average_kernel(kernel_size, device, channels):
    kernel = torch.ones(kernel_size, kernel_size, device=device) / (kernel_size * kernel_size)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    return kernel


def motion_kernel(kernel_size, angle, device, channels):
    # 构造初始运动核：沿主对角线取值1
    kernel = torch.zeros(kernel_size, kernel_size, device=device)
    for i in range(kernel_size):
        kernel[i, i] = 1.0
    kernel = kernel / kernel.sum()
    # 将 kernel 转换为 (1, 1, H, W)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    # 构造旋转矩阵
    theta = torch.tensor(
        [
            [torch.cos(torch.deg2rad(torch.tensor(angle))), -torch.sin(torch.deg2rad(torch.tensor(angle))), 0],
            [torch.sin(torch.deg2rad(torch.tensor(angle))), torch.cos(torch.deg2rad(torch.tensor(angle))), 0],
        ],
        device=device,
    ).unsqueeze(0)
    grid = F.affine_grid(theta, kernel.size(), align_corners=False)
    kernel_rotated = F.grid_sample(kernel, grid, align_corners=False, mode="bilinear")
    kernel_rotated = kernel_rotated.squeeze(0)  # (1, kernel_size, kernel_size)
    kernel_rotated = kernel_rotated.expand(channels, 1, kernel_size, kernel_size)
    return kernel_rotated


def combined_blur_batch(
    batch,
    gaussian_params={"kernel_size": 5, "sigma": 1.0},
    average_params={"kernel_size": 5},
    weights={"gaussian": 1.0, "average": 1.0, "motion": 1.0},
):
    """
    对 batch 内的所有图像同时应用高斯模糊、平均模糊和运动模糊，
    并按照指定权重融合各模糊结果。

    参数:
      batch: Tensor, shape (B, C, H, W)
      gaussian_params: dict, 包含 'kernel_size' 和 'sigma'
      average_params: dict, 包含 'kernel_size'
      motion_params: dict, 包含 'kernel_size' 和 'angle'
      weights: dict, 每种滤波器的权重
    返回:
      combined_blur: 融合后的图像, shape (B, C, H, W)
    """
    B, C, H, W = batch.shape
    device = batch.device

    # 高斯模糊
    k_size_g = gaussian_params.get("kernel_size", 5)
    sigma = gaussian_params.get("sigma", 1.0)
    kernel_gaussian = gaussian_kernel(k_size_g, sigma, device, C)
    padding_g = k_size_g // 2
    blurred_gaussian = F.conv2d(batch, kernel_gaussian, padding=padding_g, groups=C)

    # 平均模糊
    k_size_a = average_params.get("kernel_size", 5)
    kernel_average = average_kernel(k_size_a, device, C)
    padding_a = k_size_a // 2
    blurred_average = F.conv2d(batch, kernel_average, padding=padding_a, groups=C)

    # 根据权重融合三个模糊结果
    weight_g = weights.get("gaussian", 1.0)
    weight_a = weights.get("average", 1.0)

    combined_blur = (weight_g * blurred_gaussian + weight_a * blurred_average) / (weight_g + weight_a)

    return combined_blur


def drfr_mutual_augment(batch, xi=0.1):
    """
    对一个 batch 内的图像进行 DRFR 数据增强，假设 batch 中相邻两张图像构成一对，
    对每对图像进行互相增强：
      - 第一幅图像使用其原有相位和 (1-γ)*amp1 + γ*amp2 混合后的幅度生成增强图像
      - 第二幅图像使用其原有相位和 (1-γ)*amp2 + γ*amp1 混合后的幅度生成增强图像
    参数:
      batch: Tensor，形状为 (B, C, H, W)，要求 B 为偶数
      xi: 混合幅度的最大范围参数，混合系数 γ ~ U(0, xi)
    返回:
      aug_batch: 互相增强后的图像 Tensor，形状与输入相同 (B, C, H, W)
    """
    B, C, H, W = batch.shape
    is_odd = B % 2 != 0
    if is_odd:
        # 将第一个样本复制到末尾，使 batch 样本数变为偶数
        batch = torch.cat([batch, batch[0:1]], dim=0)
        B += 1  # 更新 B

    # 分离相邻图像对
    images1 = batch[::2]  # 第一张图像（用来保持其相位）
    images2 = batch[1::2]  # 第二张图像

    # 对两组图像分别进行二维傅里叶变换
    fft1 = torch.fft.fft2(images1)
    fft2 = torch.fft.fft2(images2)

    # 提取幅度和相位
    amp1 = torch.abs(fft1)
    phase1 = torch.angle(fft1)
    amp2 = torch.abs(fft2)
    phase2 = torch.angle(fft2)

    # 为每个图像对生成随机混合系数 γ，形状为 (B/2, 1, 1, 1)
    gamma = torch.rand(amp1.shape[0], 1, 1, 1, device=batch.device) * xi

    # 互相混合幅度：
    # 第一幅图像：保留其相位，混合幅度为 (1-γ)*amp1 + γ*amp2
    mixed_amp1 = (1 - gamma) * amp1 + gamma * amp2
    # 第二幅图像：保留其相位，混合幅度为 (1-γ)*amp2 + γ*amp1
    mixed_amp2 = (1 - gamma) * amp2 + gamma * amp1

    # 利用各自的相位和混合后的幅度重构复数傅里叶系数
    fft_aug1 = torch.polar(mixed_amp1, phase1)
    fft_aug2 = torch.polar(mixed_amp2, phase2)

    # 进行逆傅里叶变换，取实部得到增强后的图像
    aug1 = torch.fft.ifft2(fft_aug1).real
    aug2 = torch.fft.ifft2(fft_aug2).real

    # 重新组合成与原 batch 相同的顺序
    aug_batch = torch.empty_like(batch)
    aug_batch[::2] = aug1
    aug_batch[1::2] = aug2
    # 如果原始 batch 为奇数，则去掉最后一个多余的样本
    if is_odd:
        aug_batch = aug_batch[:-1]

    return aug_batch
