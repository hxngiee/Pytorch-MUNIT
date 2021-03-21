import torch
import torch.nn as nn
import torch.nn.functional as F

# 왜 패딩을 이렇게 분리했지
# Resnet에서도 그러네
# class Conv2dBlock(nn.Module):
#     def __init__(self, nch_in, nch_out, kernel_size, stride, padding, norm='inorm', relu=0.0, padding_mode='zeros', bias=True):
#         super(Conv2dBlock, self).__init__()
#
#         layer = []
#
#         if padding_mode != None:
#             layer += [Padding(padding=padding, padding_mode=padding_mode)]
#
#         # 아니 MUNIT 원래 Conv2d에서 코드에서 왜 padding 지웠지
#         layer += [Conv2d(nch_in,nch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=[])]
#
#         if norm != None:
#             layer += [Norm2d(nch_out, norm)]
#
#         if relu != None:
#             layer += [ReLU(relu)]
#
#         self.cbr = nn.Sequential(*layer)
#
#     def forward(self,x):
#         return self.cbr(x)

class Conv2dBlock(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size, stride, padding, norm='inorm', relu=0.0, padding_mode='zeros', bias=True, pono=False, ms=False, spectral_norm=False):
        super(Conv2dBlock, self).__init__()

        ## Pono
        affine = not ms
        # affine = True

        ## layer
        # layer = []

        self.pad = Padding(padding=padding, padding_mode=padding_mode) if padding_mode else None
        # layer += [Padding(padding=padding, padding_mode=padding_mode)]

        # 아니 MUNIT 원래 Conv2d에서 코드에서 왜 padding 지웠지
        if spectral_norm:
            self.conv = SpectralNorm(nn.Conv2d(nch_in, nch_out, kernel_size, stride, padding, bias))
        else:
            self.conv = Conv2d(nch_in,nch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=[])

        self.norm = Norm2d(nch_out,norm,affine) if norm else None

        self.relu = ReLU(relu) if relu else None

        self.pono = PONO(affine=False) if pono else None
        self.ms = MS() if ms else None

    # def forward(self,x):
    def forward(self,x, beta=None, gamma=None):
        x = self.conv(self.pad(x))
        mean, std = None, None
        if self.pono:
            x, mean, std = self.pono(x)
        if self.norm:
            x = self.norm(x)
        if self.ms:
            x = self.ms(x, beta, gamma)
        if self.relu:
            x = self.relu(x)
        if mean is None:
            return x
        else:
            return x, mean, std
        # return self.cbr(x)


class LinearBlock(nn.Module):
    def __init__(self, nch_in, nch_out, norm=None, relu=None):
        super(LinearBlock, self).__init__()

        layer = []

        layer += [nn.Linear(nch_in,nch_out,bias=True)]

        if norm != None:
            layer += [Norm2d(nch_out,norm)]

        if relu != None:
            layer += [ReLU(relu)]

        self.lbr = nn.Sequential(*layer)

    def forward(self,x):
        return self.lbr(x)

##
class CNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class DECNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Deconv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.decbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.decbr(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='inorm', relu=0.0, padding_mode='zeros'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, relu=0.0, padding_mode=padding_mode)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, relu=None, padding_mode=padding_mode)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResBlocks(nn.Module):
    def __init__(self, nch_ker, norm='inorm', relu=0.0, padding_mode='zero', nblk=4):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(nblk):
            # self.model += [ResBlock(dim, norm=norm, relu=activation, padding_mode=pad_type)]
            self.model += [ResBlock(nch_ker, norm=norm, relu=0.0, padding_mode=padding_mode)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

# class ResBlock(nn.Module):
#     def __init__(self, nch_in, nch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm='inorm', relu=0.0, drop=[], bias=[]):
#         super().__init__()
#
#         if bias == []:
#             if norm == 'bnorm':
#                 bias = False
#             else:
#                 bias = True
#
#         layers = []
#
#         # 1st conv
#         layers += [Padding(padding, padding_mode=padding_mode)]
#         layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=relu)]
#
#         if drop != []:
#             layers += [nn.Dropout2d(drop)]
#
#         # 2nd conv
#         layers += [Padding(padding, padding_mode=padding_mode)]
#         layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=[])]
#
#         self.resblk = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return x + self.resblk(x)



class CNR1d(nn.Module):
    def __init__(self, nch_in, nch_out, norm='bnorm', relu=0.0, drop=[]):
        super().__init__()

        if norm == 'bnorm':
            bias = False
        else:
            bias = True

        layers = []
        layers += [nn.Linear(nch_in, nch_out, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True):
        super(Conv2d, self).__init__()
        # self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Deconv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, bias=True):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

        # layers = [nn.Upsample(scale_factor=2, mode='bilinear'),
        #           nn.ReflectionPad2d(1),
        #           nn.Conv2d(nch_in , nch_out, kernel_size=3, stride=1, padding=0)]
        #
        # self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv(x)


class Linear(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(nch_in, nch_out)

    def forward(self, x):
        return self.linear(x)

## origin Norm Code
# class Norm2d(nn.Module):
#     def __init__(self, nch, norm_mode):
#         super(Norm2d, self).__init__()
#         if norm_mode == 'bnorm':
#             self.norm = nn.BatchNorm2d(nch)
#         elif norm_mode == 'inorm':
#             self.norm = nn.InstanceNorm2d(nch)
#         elif norm_mode == 'lnorm':
#             self.norm = LayerNorm(nch)
#         elif norm_mode == 'adain':
#             self.norm = AdaptiveInstanceNorm2d(nch)
#         elif norm_mode == 'none' or norm_mode == None:
#             self.norm = None
#
#     def forward(self, x):
#         if self.norm:
#             return self.norm(x)
#         # return self.norm(x)

class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode, affine=None):
        super(Norm2d, self).__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch,affine=affine)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch,affine=affine)
        elif norm_mode == 'lnorm':
            self.norm = LayerNorm(nch,affine=affine)
        elif norm_mode == 'adain':
            self.norm = AdaptiveInstanceNorm2d(nch)
        elif norm_mode == 'none' or norm_mode == None:
            self.norm = None

    def forward(self, x):
        if self.norm:
            return self.norm(x)
        # return self.norm(x)

# Activation으로 Naming 변경 필요
class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu == 'tanh':
            self.relu = nn.Tanh()
        elif relu == 'prelu':
            self.relu = nn.PReLU()
        elif relu == 'selu':
            self.relu = nn.SELU(True)
        elif relu == 'none':
            self.relu = None
        elif relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)


    def forward(self, x):
        return self.relu(x)


class Padding(nn.Module):
    def __init__(self, padding, padding_mode='zeros', value=0):
        super(Padding, self).__init__()
        if padding_mode == 'reflection':
            self.padding = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replication':
            self.padding = nn.ReplicationPad2d(padding)
        elif padding_mode == 'constant':
            self.padding = nn.ConstantPad2d(padding, value)
        elif padding_mode == 'zeros':
            self.padding = nn.ZeroPad2d(padding)

    def forward(self, x):
        return self.padding(x)

class Pooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='avg'):
        super().__init__()

        if type == 'avg':
            self.pooling = nn.AvgPool2d(pool)
        elif type == 'max':
            self.pooling = nn.MaxPool2d(pool)
        elif type == 'conv':
            self.pooling = nn.Conv2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pooling(x)


class UnPooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='nearest'):
        super().__init__()

        if type == 'nearest':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='nearest', align_corners=True)
        elif type == 'bilinear':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='bilinear', align_corners=True)
        elif type == 'conv':
            self.unpooling = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.unpooling(x)


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])

        return torch.cat([x2, x1], dim=1)


class TV1dLoss(nn.Module):
    def __init__(self):
        super(TV1dLoss, self).__init__()

    def forward(self, input):
        # loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
        #        torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        loss = torch.mean(torch.abs(input[:, :-1] - input[:, 1:]))

        return loss


class TV2dLoss(nn.Module):
    def __init__(self):
        super(TV2dLoss, self).__init__()

    def forward(self, input):
        loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
               torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        return loss


class SSIM2dLoss(nn.Module):
    def __init__(self):
        super(SSIM2dLoss, self).__init__()

    def forward(self, input, targer):
        loss = 0
        return loss

class GradientPaneltyLoss(nn.Module):
    def __init__(self):
        super(GradientPaneltyLoss, self).__init__()

    def forward(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones_like(y)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        _, label = target.max(dim=1)
        return self.loss(input, label.long())

##

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

#####################################################################################################################
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class PONO(nn.Module):
    def __init__(self, input_size=None, return_stats=False, affine=False, eps=1e-5):
        super(PONO, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x =  x * self.gamma + self.beta
        return x, mean, std

class MS(nn.Module):
    def __init__(self, beta=None, gamma=None):
        super(MS, self).__init__()
        self.gamma, self.beta = gamma, beta

    def forward(self, x, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x

## model 파트로 이동
class PONOContentEncoder(nn.Module):
    def __init__(self, nch_in, nch_ker, norm, relu, padding_mode, nblk, n_down, pono=True):
        super(PONOContentEncoder, self).__init__()
        self.pono = pono
        self.ds_blocks = nn.ModuleList()

        #원본 self.ds_blocks.append(Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type, pono=pono))
        self.ds_blocks.append(Conv2dBlock(nch_in=nch_in, nch_out=nch_ker, kernel_size=7, stride=1, padding=3, norm=norm, relu=relu, padding_mode=padding_mode, pono=pono))

        # downsampling blocks
        for i in range(n_down):
            self.ds_blocks.append(Conv2dBlock(nch_in=nch_ker, nch_out=2 * nch_ker, kernel_size=4, stride=2, padding=1, norm=norm, relu=relu, padding_mode=padding_mode, pono=pono))
            nch_ker *= 2
        # residual blocks
        # self.resblocks = ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)
        self.resblocks = ResBlocks(nch_ker, norm=norm, relu=relu, padding_mode=padding_mode, nblk=nblk)

        self.output_dim = nch_ker

    def forward(self, x):
        stats = []
        for block in self.ds_blocks:
            if self.pono:
                x, mean, std = block(x)
                stats.append((mean, std))
            else:
                x = block(x)
        x = self.resblocks(x)
        return x, stats
