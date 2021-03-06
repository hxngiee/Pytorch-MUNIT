from layer import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

class Generator(nn.Module):
    def __init__(self, nch_in, nch_ker, norm = None, relu=0.0, padding_mode='reflection'):
        super(Generator, self).__init__()
        self.nch_ker = nch_ker
        self.style_dim = 8
        self.nblk = 4
        self.mlp_dim = 256

        pono = True

        # Style Encoder
        self.enc_style = StyleEncoder(nch_in=nch_in, nch_ker=nch_ker, nch_sty=self.style_dim, norm=None)

        # Content Encoder
        ## PONO + ContentEncoder
        if pono:
            self.enc_content = PONOContentEncoder(nch_in=nch_in, nch_ker=nch_ker, norm='inorm', relu=0.2, padding_mode=padding_mode, nblk=4, n_down=2, pono=pono)
            self.dec = MSDecoder(nch_in=self.enc_content.output_dim, nch_out=3, norm='adain', relu=0.2, padding_mode=padding_mode)
        else:
            ##### Decoder 파트 이상
            # self.enc_content.output_dim 256
            self.enc_content = ContentEncoder(nch_in=nch_in, nch_ker=nch_ker, norm='inorm', padding_mode=padding_mode, nblk=4, n_down=2)
            self.dec = Decoder(nch_in=self.enc_content.output_dim, nch_out=3, norm='adain', relu=0.2,padding_mode=padding_mode)

        # MLP to generate AdaIN params
        self.mlp = MLP(nch_in=self.style_dim, nch_out=self.get_num_adain_params(self.dec), nch_ker=self.mlp_dim, nblk=3, norm=None, relu=relu)

        ## functions for PONO
        if pono:
            self.stat_convs = nn.ModuleList()

            net = nn.Sequential(Conv2dBlock(nch_in=2,nch_out=self.nch_ker, kernel_size=3, stride=1, padding=1, norm='bnorm',relu=0.0,padding_mode='zeros'),
                                Conv2dBlock(self.nch_ker,2*self.nch_ker,3,1,1,norm=None,relu=None,padding_mode='zeros'))
            self.stat_convs.append(net)
            n_down = 2 # number of downsample
            for i in range(n_down):
                net = []
                net = nn.Sequential(*net)
                self.stat_convs.append(net)

    def forward(self, x):
        content, style_fake = self.encode(x)
        rec_x = self.decode(content, style_fake)
        return rec_x

    def encode(self, x):
        style_fake = self.enc_style(x)
        content = self.enc_content(x)
        return content, style_fake

    # def decode(self, content, style):
    #     adain_params = self.mlp(style)
    #     self.assign_adain_params(adain_params, self.dec)
    #     x = self.dec(content)
    #     return x

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)

        new_stats = [stat for stat in content[1]]
        new_stats.reverse()

        self.assign_adain_params(adain_params, self.dec)
        x = self.dec(content[0], new_stats)
        # x = self.dec(content)
        return x

    def assign_adain_params(self, adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIn params needed by the model
        num_adain_params = 0

        for m in model.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                num_adain_params += 2 * m.num_features

        return num_adain_params


class Discriminator(nn.Module):
    # Multi-scale Discriminator
    def __init__(self, nch_in, nch_ker, norm=None, relu=0.0, padding_mode='reflection'):
        super(Discriminator, self).__init__()
        self.nch_in = nch_in
        self.nch_ker = nch_ker

        self.n_layer = 4

        self.norm = norm
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1,1], count_include_pad=False)
        # self.activ = relu # lrelu 씀
        # self.loss # GAN Loss[lsgan/nsgan]

        # nn.Sequential과 nn.Modullist 차이
        self.cnns = nn.ModuleList()

        for _ in range(3):
            self.cnns.append(self.make_net())

    def make_net(self):
        nch_ker = self.nch_ker
        dk_layer = []
        dk_layer += [Conv2dBlock(self.nch_in, nch_ker, 4, 2, 1, norm=None, relu=None, padding_mode='reflection')]
        for i in range(self.n_layer - 1):
            dk_layer += [Conv2dBlock(nch_ker, 2 * nch_ker,4, 2, 1, norm=self.norm, relu=0.2, padding_mode='reflection')]
            nch_ker *= 2
        dk_layer = nn.Sequential(*dk_layer)
        return dk_layer

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


class ContentEncoder(nn.Module):
    def __init__(self,nch_in, nch_ker, norm, padding_mode, nblk=4, n_down=2):
        super(ContentEncoder, self).__init__()

        self.model = []
        self.model += [Conv2dBlock(nch_in,nch_ker,7,1,3, norm=norm, relu=0.0, padding_mode=padding_mode)]

        # downsampling blocks
        for i in range(n_down):
            self.model += [Conv2dBlock(nch_ker, 2 * nch_ker, 4, 2, 1, norm=norm, relu=0.0, padding_mode=padding_mode)]
            nch_ker *= 2

        # residual blocks
        self.model += [ResBlocks(nblk, nch_ker, norm=norm, activation='relu', pad_type='reflection')]
        # for i in range(nblk):
        #     self.model += [ResBlock(nch_ker, nch_ker,kernel_size=3, stride=1, padding=1, norm=norm, relu=0.0, padding_mode=padding_mode)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = nch_ker     # decoder output_dim과 연결

    def forward(self,x):

        return self.model(x)

class StyleEncoder(nn.Module):
    def __init__(self, nch_in, nch_ker, nch_sty=8, norm=None):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(nch_in, nch_ker, 7, 1, 3, norm=norm, relu=0.0)]

        for i in range(2):
            self.model += [Conv2dBlock(nch_ker, 2 * nch_ker, 4, 2, 1, norm=norm, relu=0.0)]
            nch_ker *= 2

        # d256, d256
        for i in range(2):
            self.model += [Conv2dBlock(nch_ker, nch_ker, 4, 2, 1, norm=norm, relu=0.0)]

        # GAP - global average pooling
        self.model += [nn.AdaptiveAvgPool2d(1)]

        # fc8
        self.model += [nn.Conv2d(nch_ker, nch_sty, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = nch_ker

    def forward(self,x):
        return self.model(x)


# n_res ->4 d_sam -> 2
class MSDecoder(nn.Module):
    def __init__(self, nch_in, nch_out, norm='adain', relu=0.0, padding_mode='zeros', ms=True):
        super(MSDecoder, self).__init__()

        # AdaIN residual blocks
        self.res_blocks = ResBlocks(nch_in, norm, relu, padding_mode=padding_mode, nblk=4)
        # upsampling blocks
        self.us_blocks = nn.ModuleList()
        for i in range(2):
            self.us_blocks.append(Conv2dBlock(nch_in, nch_in // 2, 5, 1, 2, norm='lnorm', relu=0.2, padding_mode=padding_mode, ms=ms))
            self.us_blocks.append(nn.Upsample(scale_factor=2))
            nch_in //= 2
        # use reflection padding in the last conv layer
        self.us_blocks.append(Conv2dBlock(nch_in, nch_in, 3, 1, 1, norm='lnorm', relu=0.2, padding_mode=padding_mode, ms=ms))
        self.last_block = Conv2dBlock(nch_in, nch_out, 7, 1, 3, norm=None, relu='tanh', padding_mode=padding_mode)

    def forward(self, x, stats):
        x = self.res_blocks(x)
        i = 0
        for block in self.us_blocks:
            if isinstance(block, Conv2dBlock):
                beta, gamma = stats[i]
                x = block(x, beta, gamma)
                i += 1
            else:
                x = block(x)
        return self.last_block(x)



class Decoder(nn.Module):
    def __init__(self, nch_in, nch_out, norm='adain', relu=None, padding_mode='zeros', nblk=4, n_up=2):
        super(Decoder, self).__init__()

        self.model = []

        # self.model += [ResBlocks(nblk, nch_in, norm, relu, padding_mode)]
        self.model += [ResBlocks(nblk, nch_in, norm, relu, pad_type=padding_mode)]

        # AdaIN residual blocks
        # for i in range(nblk):
        #     self.model += [ResBlock(nch_in, nch_in,kernel_size=3, stride=1, padding=1, norm=norm, relu=0.0, padding_mode=padding_mode)]
        #     self.model += [ResBlock(nch_in, nch_in,kernel_size=3, stride=1, padding=1, norm=norm, relu=[], padding_mode=padding_mode)]

        # Upsample
        for i in range(n_up):
            self.model += [nn.Upsample(scale_factor=2)]
            self.model += [Conv2dBlock(nch_in, nch_in//2, 5, 1, 2, norm='lnorm', relu=0.0, padding_mode=padding_mode)]

            nch_in = nch_in//2

        # relu 파트 수정 필요
        self.model += [Conv2dBlock(nch_in,nch_out, 7, 1, 3, norm=None, relu='tanh', padding_mode=padding_mode)]
        self.model = nn.Sequential(*self.model)

    def forward(self,x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=256, nblk=3, norm=None, relu=0.0):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(nch_in, nch_ker, norm=norm, relu=relu)]
        for i in range(nblk-2):
            self.model += [LinearBlock(nch_ker,nch_ker, norm=norm, relu=relu)]
        self.model += [LinearBlock(nch_ker, nch_out, norm=None, relu=None)]
        self.model = nn.Sequential(*self.model)
    def forward(self,x):
        return self.model(x.view(x.size(0), -1))



##
class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc5 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc6 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc7 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc8 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])

        self.dec8 = DECNR2d(1 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=0.5)
        self.dec7 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=0.5)
        self.dec6 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=0.5)
        self.dec5 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec4 = DECNR2d(2 * 8 * self.nch_ker, 4 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec3 = DECNR2d(2 * 4 * self.nch_ker, 2 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec2 = DECNR2d(2 * 2 * self.nch_ker, 1 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec1 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_out, stride=2, norm=[],        relu=[],  drop=[], bias=False)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)

        dec8 = self.dec8(enc8)
        dec7 = self.dec7(torch.cat([enc7, dec8], dim=1))
        dec6 = self.dec6(torch.cat([enc6, dec7], dim=1))
        dec5 = self.dec5(torch.cat([enc5, dec6], dim=1))
        dec4 = self.dec4(torch.cat([enc4, dec5], dim=1))
        dec3 = self.dec3(torch.cat([enc3, dec4], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dec1(torch.cat([enc1, dec2], dim=1))

        x = torch.tanh(dec1)

        return x

class ResNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=6):
        super(ResNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(self.nch_in,      1 * self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)

        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        if self.nblk:
            res = []

            for i in range(self.nblk):
                res += [ResBlock(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]

            self.res = nn.Sequential(*res)

        self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec2 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec1 = CNR2d(1 * self.nch_ker, self.nch_out, kernel_size=7, stride=1, padding=3, norm=[], relu=[], bias=False)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        if self.nblk:
            x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

