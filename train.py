from model import *
from dataset import *

# import vgg function, self.opts from utils
from utils import *

import itertools

from statistics import mean

import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        # scheduler mode
        # 이거 main으로부터 깔끔히 처리하는 방법 생각해보기 직접상속 안받고
        self.opts = sched_opts()

        self.wgt_gan = args.wgt_gan
        self.wgt_rec_x = args.wgt_rec_x
        self.wgt_rec_x_cyc = args.wgt_rec_x_cyc
        self.wgt_rec_s = args.wgt_rec_s
        self.wgt_rec_c = args.wgt_rec_c
        self.wgt_vgg = args.wgt_vgg

        self.optim = args.optim
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.name_data = args.name_data

        self.nblk = args.nblk

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG, netD, optimG, optimD, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG, netD, optimG=[], optimD=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            netD.load_state_dict(dict_net['netD'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])

            return netG, netD, optimG, optimD, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])

            return netG, epoch

    def preprocess(self, data):
        normalize = Normalize()
        randflip = RandomFlip()
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_out, self.nx_out))
        totensor = ToTensor()
        return totensor(randomcrop(rescale(randflip(normalize(data)))))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()
        return denomalize(tonumpy(data))

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_gan = self.wgt_gan
        wgt_rec_x = self.wgt_rec_x
        wgt_rec_x_cyc = self.wgt_rec_x_cyc
        wgt_rec_s = self.wgt_rec_s
        wgt_rec_c = self.wgt_rec_c
        wgt_vgg = self.wgt_vgg

        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data)

        dir_log_train = os.path.join(self.dir_log, self.scope, name_data)

        transform_train = transforms.Compose([CenterCrop((self.ny_load, self.nx_load)), Normalize(), RandomFlip(), Rescale((self.ny_in, self.nx_in)), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_train = Dataset(data_dir=os.path.join(dir_data_train,'train'), data_type=self.data_type, transform=transform_train)

        # loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

        num_train = len(dataset_train)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        netG_a2b = Generator(nch_in=nch_in, nch_ker=nch_ker, relu=0.0, padding_mode='reflection')
        netG_b2a = Generator(nch_in=nch_in, nch_ker=nch_ker, relu=0.0, padding_mode='reflection')
        netD_a = Discriminator(nch_in=nch_in, nch_ker=nch_ker)
        netD_b = Discriminator(nch_in=nch_in, nch_ker=nch_ker)

    # kaiming init 하는 듯
        init_net(netG_a2b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netG_b2a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD_a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD_b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## fix the noise used in sampling


        ## setup loss & optimization
        fn_REC = nn.L1Loss().to(device)   # L1
        # fn_SRC = nn.BCEWithLogitsLoss().to(device)
        # fn_GP = GradientPaneltyLoss().to(device)

        paramsG_a = netG_a2b.parameters()
        paramsG_b = netG_b2a.parameters()
        paramsD_a = netD_a.parameters()
        paramsD_b = netD_b.parameters()

        optimG_a = torch.optim.Adam(paramsG_a, lr=lr_G, betas=(self.beta1, self.beta2), weight_decay=0.0001)
        optimG_b = torch.optim.Adam(paramsG_b, lr=lr_G, betas=(self.beta1, self.beta2), weight_decay=0.0001)
        optimD_a = torch.optim.Adam(paramsD_a, lr=lr_D, betas=(self.beta1, self.beta2), weight_decay=0.0001)
        optimD_b = torch.optim.Adam(paramsD_b, lr=lr_D, betas=(self.beta1, self.beta2), weight_decay=0.0001)

        schedG_a = get_scheduler(optimG_a, self.opts)
        schedG_b = get_scheduler(optimG_b, self.opts)
        schedD_a = get_scheduler(optimD_a, self.opts)
        schedD_b = get_scheduler(optimD_b, self.opts)

        schedG_a = torch.optim.lr_scheduler.ExponentialLR(optimG_a, gamma=0.9)
        schedG_b = torch.optim.lr_scheduler.ExponentialLR(optimG_b, gamma=0.9)
        schedD_a = torch.optim.lr_scheduler.ExponentialLR(optimD_a, gamma=0.9)
        schedD_b = torch.optim.lr_scheduler.ExponentialLR(optimD_b, gamma=0.9)

        ## load from checkpoints
        st_epoch = 0

        # 이 파트 수정 필요
        # if train_continue == 'on':
        #     netG, netD, optimG, optimD, st_epoch = self.load(dir_chck, netG, netD, optimG, optimD, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG_a2b.train()
            netG_b2a.train()
            netD_a.train()
            netD_b.train()

            loss_D_a_train = []
            loss_D_b_train = []

            loss_G_a2b_train = []
            loss_G_b2a_train = []
            loss_G_rec_a_train = []
            loss_G_rec_b_train = []
            loss_G_rec_s_a_train = []
            loss_G_rec_s_b_train = []
            loss_G_rec_c_a_train = []
            loss_G_rec_c_b_train = []
            loss_G_rec_x_a_train = []
            loss_G_rec_x_b_train = []

            for i, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                # Copy to GPU
                input_a = data['data_a'].to(device)
                input_b = data['data_b'].to(device)

                # backward netD
                set_requires_grad(netD_a, True)
                set_requires_grad(netD_b, True)

                optimD_a.zero_grad()
                optimD_b.zero_grad()

                # style.dim = 8
                s_a = Variable(torch.randn(input_a.size(0), 8, 1, 1).to(device))
                s_b = Variable(torch.randn(input_b.size(0), 8, 1, 1).to(device))

                # forward
                # encode
                c_a, _ = netG_b2a.encode(input_a)
                c_b, _ = netG_a2b.encode(input_b)

                # decode (Cross domain)
                output_a = netG_b2a.decode(c_b, s_a)
                output_b = netG_a2b.decode(c_a, s_b)

                #pred_fake_a ->outs0,  pred_real_a ->outs1
                pred_real_a = netD_a.forward(input_a)
                pred_fake_a = netD_a.forward(output_a.detach())

                for out0, out1 in zip(pred_fake_a,pred_real_a):
                    loss_D_a = 0.5 * (torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2))

                pred_real_b = netD_b.forward(input_b)
                pred_fake_b = netD_b.forward(output_b.detach())

                for out0, out1 in zip(pred_fake_b,pred_real_b):
                    loss_D_b = 0.5 * (torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2))

                # total loss
                loss_D = loss_D_a * wgt_gan + loss_D_b * wgt_gan

                loss_D.backward()
                optimD_a.step()
                optimD_b.step()


                ## Generator
                set_requires_grad(netD_a, False)
                set_requires_grad(netD_b, False)

                optimG_a.zero_grad()
                optimG_b.zero_grad()

                # style.dim = 8
                s_a = Variable(torch.rand(input_a.size(0) ,8, 1, 1).to(device))
                s_b = Variable(torch.rand(input_b.size(0) ,8, 1, 1).to(device))

                # encode
                c_a, s_a_prime = netG_a2b.encode(input_a)
                c_b, s_b_prime = netG_b2a.encode(input_b)

                # decode (within domain)
##########decode파트 이상
                recon_a = netG_a2b.decode(c_a, s_a_prime)
                recon_b = netG_b2a.decode(c_b, s_b_prime)

                # decode (cross domain)
                output_a = netG_b2a.decode(c_b,s_a)
                output_b = netG_a2b.decode(c_a,s_b)

                # encode again
                recon_c_b, recon_s_a = netG_b2a.encode(output_a)
                recon_c_a, recon_s_b = netG_a2b.encode(output_b)

                # decode again
                ident_a = netG_b2a.decode(recon_c_a, s_a_prime)
                ident_b = netG_a2b.decode(recon_c_b, s_b_prime)

                # loss_G
                pred_fake_a = netD_a.forward(output_a)
                pred_fake_b = netD_b.forward(output_b)

                for out0 in pred_fake_a:
                    loss_G_a2b = torch.mean((out0 - 1)**2)
                for out0 in pred_fake_b:
                    loss_G_b2a = torch.mean((out0 - 1)**2)

                loss_G_rec_a = fn_REC(recon_a, input_a)
                loss_G_rec_b = fn_REC(recon_b, input_b)
                loss_G_rec_s_a = fn_REC(recon_s_a, s_a)
                loss_G_rec_s_b = fn_REC(recon_s_a, s_b)
                loss_G_rec_c_a = fn_REC(recon_c_a,c_a)
                loss_G_rec_c_b = fn_REC(recon_c_b,c_b)
                loss_G_rec_x_a = fn_REC(ident_a, input_a)
                loss_G_rec_x_b = fn_REC(ident_b, input_b)

                # loss_G_vgg_a = compute_vgg_loss(vgg추가해야함,output_a, input_b)
                # loss_G_vgg_b = compute_vgg_loss(vgg추가해야함,output_b, input_a)


                loss_G = wgt_gan * loss_G_a2b + wgt_gan * loss_G_b2a + \
                         wgt_rec_x * loss_G_rec_a + wgt_rec_x * loss_G_rec_b + \
                         wgt_rec_c * loss_G_rec_c_a + wgt_rec_c * loss_G_rec_c_b + \
                         wgt_rec_s * loss_G_rec_s_a + wgt_rec_s * loss_G_rec_s_b + \
                         wgt_rec_x_cyc * loss_G_rec_x_a + wgt_rec_x_cyc * loss_G_rec_x_b
                         # wgt_vgg * loss_G_vgg_a + wgt_vgg * loss_G_vgg_b


                loss_G.backward()
                optimG_a.step()
                optimG_b.step()

                # get losses
                loss_D_a_train += [loss_D_a.item()]
                loss_D_b_train += [loss_D_b.item()]


                loss_G_a2b_train += [loss_G_a2b.item()]
                loss_G_b2a_train += [loss_G_b2a.item()]
                loss_G_rec_a_train += [loss_G_rec_a.item()]
                loss_G_rec_b_train += [loss_G_rec_b.item()]
                loss_G_rec_s_a_train += [loss_G_rec_s_a.item()]
                loss_G_rec_s_b_train += [loss_G_rec_s_b.item()]
                loss_G_rec_c_a_train += [loss_G_rec_c_a.item()]
                loss_G_rec_c_b_train += [loss_G_rec_c_b.item()]
                loss_G_rec_x_a_train += [loss_G_rec_x_a.item()]
                loss_G_rec_x_b_train += [loss_G_rec_x_b.item()]


                print('TRAIN: EPOCH %d: BATCH %03d/%04d: '
                      "GEN a2b %.4f b2a %.4f | "
                      "DISC a %.4f b %.4f | "
                      "CYCLE a %.4f b %.4f | "
                      "IDENT a %.4f b %.4f | "
                      "RECON_S a %.4f b %.4f | "
                      "RECON_C a %.4f b %.4f | " % (epoch, i, num_batch_train,
                                                   mean(loss_G_a2b_train), mean(loss_G_b2a_train), mean(loss_D_a_train), mean(loss_D_b_train),
                                                   mean(loss_G_rec_a_train), mean(loss_G_rec_b_train), mean(loss_G_rec_x_a_train), mean(loss_G_rec_x_b_train),
                                                   mean(loss_G_rec_s_a_train),mean(loss_G_rec_s_b_train),mean(loss_G_rec_c_a_train), mean(loss_G_rec_c_b_train)))

                if should(num_freq_disp):
                    ## show output
                    input_a = transform_inv(input_a)
                    input_b = transform_inv(input_b)
                    output_a = transform_inv(output_a)
                    output_b = transform_inv(output_b)
                    # recon = transform_inv(recon)
                    # recon = transform_inv(recon)

                    writer_train.add_images('input_a', input_a, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('input_b', input_b, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output_a', output_a, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output_b', output_b, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    # writer_train.add_images('recon', recon, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            writer_train.add_scalar('loss_D_a', mean(loss_D_a_train), epoch)
            writer_train.add_scalar('loss_D_b', mean(loss_D_b_train), epoch)
            writer_train.add_scalar('loss_G_a2b', mean(loss_G_a2b_train), epoch)
            writer_train.add_scalar('loss_G_b2a', mean(loss_G_b2a_train), epoch)
            writer_train.add_scalar('loss_G_rec_a_train', mean(loss_G_rec_a_train), epoch)
            writer_train.add_scalar('loss_G_rec_b_train', mean(loss_G_rec_b_train), epoch)
            writer_train.add_scalar('loss_G_rec_x_a_train', mean(loss_G_rec_x_a_train), epoch)
            writer_train.add_scalar('loss_G_rec_x_b_train', mean(loss_G_rec_x_b_train), epoch)
            writer_train.add_scalar('loss_G_rec_s_a_train', mean(loss_G_rec_s_a_train), epoch)
            writer_train.add_scalar('loss_G_rec_s_b_train', mean(loss_G_rec_s_b_train), epoch)
            writer_train.add_scalar('loss_G_rec_c_a_train', mean(loss_G_rec_c_a_train), epoch)
            writer_train.add_scalar('loss_G_rec_c_b_train', mean(loss_G_rec_c_b_train), epoch)

            # update schduler
            schedG_a.step()
            schedG_b.step()
            schedD_a.step()
            schedD_b.step()

## Save파트 재작성 필요
            ## save
            # if (epoch % num_freq_save) == 0:
            #     self.save(dir_chck, netG, netD, optimG, optimD, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        # batch_size = self.batch_size
        batch_size = 1
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        attrs = self.attrs
        ncls = self.ncls

        ncritic = self.ncritic

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_test = os.path.join(self.dir_data, name_data)

        dir_result = os.path.join(self.dir_result, self.scope, name_data)
        dir_result_save = os.path.join(dir_result, 'images')
        if not os.path.exists(dir_result_save):
            os.makedirs(dir_result_save)

        # dir_data_test = os.path.join(self.dir_data, self.name_data, 'test')

        # transform_test = transforms.Compose([Normalize(), ToTensor()])
        transform_test = transforms.Compose([CenterCrop((self.ny_load, self.nx_load)), Normalize(), RandomFlip(), Rescale((self.ny_in, self.nx_in)), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_test = Dataset(dir_data_test, data_type=self.data_type, transform=transform_test, attrs=attrs, mode='test')

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_test = len(dataset_test)

        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        # netG = UNet(nch_in + ncls, nch_out, nch_ker, norm)
        netG = ResNet(nch_in + ncls, nch_out, nch_ker, norm, nblk=self.nblk)
        netD = Discriminator(nch_out, nch_ker, [], ncls=ncls, ny_in=self.ny_out, nx_in=self.nx_out)

        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## load from checkpoints
        st_epoch = 0

        netG, st_epoch = self.load(dir_chck, netG, netD, mode=mode)

        ## test phase
        with torch.no_grad():
            netG.eval()
            # netG.train()

            for i, data in enumerate(loader_test, 1):
                # input = data[0]
                # label_in = data[1].view(-1, ncls, 1, 1)
                # label_out = torch.zeros_like(label_in)

                fileset = {}
                fileset['name'] = i

                for j, attr in enumerate(self.attrs):
                    input = data[0]
                    label_in = data[1].view(-1, ncls, 1, 1)
                    label_out = torch.zeros_like(label_in)
                    label_out[:, j, :, :] = 1

                    domain_out = get_domain(input, label_out)

                    # Copy to GPU
                    input = input.to(device)
                    domain_out = domain_out.to(device)
                    label_out = label_out.to(device)

                    output = netG(torch.cat([input, domain_out], dim=1))

                    input = transform_inv(input)
                    output = transform_inv(output)

                    if j == 0:
                        fileset['input'] = '%04d-input.png' % i
                        plt.imsave(os.path.join(dir_result_save, fileset['input']), input.squeeze())

                    fileset[attr] = '%04d-output-%s.png' % (i, attr)
                    plt.imsave(os.path.join(dir_result_save, fileset[attr]), output.squeeze())

                append_index(dir_result, fileset)

                print("%d / %d" % (i, num_test))


def get_domain(input, label):
    domain = label.clone()
    domain = domain.repeat(1, 1, input.size(2), input.size(3))

    return domain


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
