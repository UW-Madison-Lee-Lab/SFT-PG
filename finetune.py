import os
import argparse
import time
from tqdm import tqdm
from torch.distributions import Normal
import torch.nn.functional as F
from data import get_dataset, fix_legacy_dict
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from critic import Discriminator, ValueCelebA, Value
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
unsqueeze3x = lambda x: x[..., None, None, None]
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from model import Model
from config import diffusion_config
from torch_ema import ExponentialMovingAverage



def _map_gpu(gpu):
    if gpu == 'cuda':
        return lambda x: x.cuda()
    else:
        return lambda x: x.to(torch.device(gpu))


def rescale(X, batch=True):
    # for plot
    # if not batch:
    #     # return X
    #     return (X - (-1)) / (2)
    #     # return (X - X.min()) / (X.max()-X.min())
    # else:
    #     for i in range(X.shape[0]):
    #         X[i] = rescale(X[i], batch=False)
    return (X - (-1)) / (2)

def rescale_train(X, batch = True):
    # if not batch:
    #     # return X
    #     return (X - X.min()) / (X.max()-X.min())
    # else:
    #     for i in range(X.shape[0]):
    #         X[i] = rescale_train(X[i], batch=False)
    return X



def std_normal(size):
    return map_gpu(torch.normal(0, 1, size=size))


def print_size(net):
    """
    Print the number of parameters of a network
    """
    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
    """

    Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t-1]
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)
    
    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def bisearch(f, domain, target, eps=1e-8):
    """
    find smallest x such that f(x) > target

    Parameters:
    f (function):               function
    domain (tuple):             x in (left, right)
    target (float):             target value
    
    Returns:
    x (float)
    """
    # 
    sign = -1 if target < 0 else 1
    left, right = domain
    for _ in range(1000):
        x = (left + right) / 2 
        if f(x) < target:
            right = x
        elif f(x) > (1 + sign * eps) * target:
            left = x
        else:
            break
    return x


def get_VAR_noise(S, schedule='linear'):
    """
    Compute VAR noise levels

    Parameters:
    S (int):            approximante diffusion process length
    schedule (str):     linear or quadratic
    
    Returns:
    np array of noise levels, size = (S, )
    """
    target = np.prod(1 - np.linspace(diffusion_config["beta_0"], diffusion_config["beta_T"], diffusion_config["T"]))

    if schedule == 'linear':
        g = lambda x: np.linspace(diffusion_config["beta_0"], x, S)
        domain = (diffusion_config["beta_0"], 0.99)
    elif schedule == 'quadratic':
        g = lambda x: np.array([diffusion_config["beta_0"] * (1+i*x) ** 2 for i in range(S)])
        domain = (0.0, 0.95 / np.sqrt(diffusion_config["beta_0"]) / S)
    else:
        raise NotImplementedError

    f = lambda x: np.prod(1 - g(x))
    largest_var = bisearch(f, domain, target, eps=1e-4)
    return g(largest_var)


def _log_gamma(x):
    # Gamma(x+1) ~= sqrt(2\pi x) * (x/e)^x  (1 + 1 / 12x)
    y = x - 1
    return np.log(2 * np.pi * y) / 2 + y * (np.log(y) - 1) + np.log(1 + 1 / (12 * y))


def _log_cont_noise(t, beta_0, beta_T, T):
    # We want log_cont_noise(t, beta_0, beta_T, T) ~= np.log(Alpha_bar[-1].numpy())
    delta_beta = (beta_T - beta_0) / (T - 1)
    _c = (1.0 - beta_0) / delta_beta
    t_1 = t + 1
    return t_1 * np.log(delta_beta) + _log_gamma(_c + 1) - _log_gamma(_c - t_1 + 1)


# VAR
def _precompute_VAR_steps(diffusion_hyperparams, user_defined_eta):
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = map_gpu(torch.from_numpy(user_defined_eta)).to(torch.float32)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    
    continuous_steps = []
    with torch.no_grad():
        for t in range(T_user-1, -1, -1):
            t_adapted = None
            for i in range(T - 1):
                if Alpha_bar[i] >= Gamma_bar[t] > Alpha_bar[i+1]:
                    t_adapted = bisearch(f=lambda _t: _log_cont_noise(_t, Beta[0].cpu().numpy(), Beta[-1].cpu().numpy(), T), 
                                            domain=(i-0.01, i+1.01), 
                                            target=np.log(Gamma_bar[t].cpu().numpy()))
                    break
            if t_adapted is None:
                t_adapted = T - 1
            continuous_steps.append(t_adapted)  # must be decreasing
    return continuous_steps


def VAR_sampling(net, size, diffusion_hyperparams, user_defined_eta, kappa, continuous_steps):
    """
    Perform the complete sampling step according to user defined variances

    Parameters:
    net (torch network):            the model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    user_defined_eta (np.array):    User defined noise       
    kappa (float):                  factor multipled over sigma, between 0 and 1
    continuous_steps (list):        continuous steps computed from user_defined_eta

    Returns:
    the generated images in torch.tensor, shape=size
    """
    # net.eval()
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T
    assert len(size) == 4
    assert 0.0 <= kappa <= 1.0

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = map_gpu(torch.from_numpy(user_defined_eta)).to(torch.float32)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    
    # print('begin sampling, total number of reverse steps = %s' % T_user)

    
    x = std_normal(size)
    x_seq = [x.detach().clone()]
    log_prob_list = []

    with torch.no_grad():
        for i, tau in enumerate(continuous_steps):
            diffusion_steps = tau * map_gpu(torch.ones(size[0]))
            # print(diffusion_steps.item())
            epsilon_theta = net(x, diffusion_steps)
            # epsilon_theta1 = net(x, diffusion_steps)
            # print(epsilon_theta[0,0,0,0] == epsilon_theta1[0,0,0,0])
            if i == T_user - 1:  # the next step is to generate x_0
                assert abs(tau) < 0.1
                alpha_next = torch.tensor(1.0) 
                sigma = torch.tensor(0.0) 
            else:
                alpha_next = Gamma_bar[T_user-1-i - 1]
                sigma = kappa * torch.sqrt((1-alpha_next) / (1-Gamma_bar[T_user-1-i]) * (1 - Gamma_bar[T_user-1-i] / alpha_next))
            x *= torch.sqrt(alpha_next / Gamma_bar[T_user-1-i]) # x_prev multiplier
            c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Gamma_bar[T_user-1-i]) * torch.sqrt(alpha_next / Gamma_bar[T_user-1-i]) # theta multiplier
            pred_mean = x + c*epsilon_theta
            if i == T_user - 1:
                x += c * epsilon_theta + 0.001 * std_normal(size)
                pred_std = map_gpu(unsqueeze3x(torch.tensor([0.001])))
            else:
                x += c * epsilon_theta + sigma * std_normal(size)
                pred_std = map_gpu(unsqueeze3x(sigma))
            dist = Normal(pred_mean, pred_std)
            log_prob = dist.log_prob(x.detach().clone()).mean(dim = -1).mean(dim = -1).mean(dim = -1)
            log_prob_list.append(log_prob)
            # pred_list.append(epsilon_theta.detach().clone())
            x_seq.append(x.detach().clone())

    return x_seq, log_prob_list

def VAR_get_params(size, diffusion_hyperparams, user_defined_eta, kappa, continuous_steps):

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T
    assert len(size) == 4
    assert 0.0 <= kappa <= 1.0

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = map_gpu(torch.from_numpy(user_defined_eta)).to(torch.float32)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    
    # print('begin sampling, total number of reverse steps = %s' % T_user)
    x_prev_multiplier = torch.zeros(T_user)
    theta_multiplier = torch.zeros(T_user)
    std = torch.zeros(T_user)
    diffusion_steps_list = torch.zeros(T_user)


    for i, tau in enumerate(continuous_steps):
        diffusion_steps_list[i] = tau
        if i == T_user - 1:  # the next step is to generate x_0
            assert abs(tau) < 0.1
            alpha_next = torch.tensor(1.0) 
            sigma = torch.tensor(0.0) 
        else:
            alpha_next = Gamma_bar[T_user-1-i - 1]
            sigma = kappa * torch.sqrt((1-alpha_next) / (1-Gamma_bar[T_user-1-i]) * (1 - Gamma_bar[T_user-1-i] / alpha_next))
        x_prev_multiplier[i] = torch.sqrt(alpha_next / Gamma_bar[T_user-1-i])
        theta_multiplier[i] = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Gamma_bar[T_user-1-i]) * torch.sqrt(alpha_next / Gamma_bar[T_user-1-i]) 
        if i == T_user - 1: 
            std[i] = 0.001
        else:
            std[i] = sigma

    # x = std_normal(size)
    # with torch.no_grad():
    #     for i, tau in enumerate(continuous_steps):
    #         diffusion_steps = tau * map_gpu(torch.ones(size[0])) # shape ([bs])
    #         epsilon_theta = net(x, diffusion_steps)
    #         x = x*x_prev_multiplier[i] + theta_multiplier[i]*epsilon_theta + std[i]*std_normal(size)
    return map_gpu(x_prev_multiplier), map_gpu(theta_multiplier), map_gpu(std), map_gpu(diffusion_steps_list)

def VAR_log_prob(net, x_prev, x_next, t, x_prev_multiplier, theta_multiplier, std, diffusion_steps_list, x_seq, log_prob_tensor):
    # net.eval()
    # net.train()
    diffusion_steps = diffusion_steps_list[t] # shape ([bs])
    epsilon_theta = net(x_prev, diffusion_steps)
    # epsilon_theta_seq = net(torch.cat(x_seq[:10]), diffusion_steps)
    pred_mean = x_prev*unsqueeze3x(x_prev_multiplier[t]) + unsqueeze3x(theta_multiplier[t])*epsilon_theta 
    pred_std = unsqueeze3x(std[t])
    dist = Normal(pred_mean, pred_std)
    log_prob = dist.log_prob(x_next.detach()).mean(dim = -1).mean(dim = -1).mean(dim = -1)

    return log_prob

def train_one_epoch(net, ema, dataloader, optimizer, f, v, optimizer_fstar, optimizer_v, continuous_steps, diffusion_hyperparams, size, user_defined_eta, kappa, args, train):
    # buffer
    state_dict = {}
    state_dict['state'] = map_gpu(torch.FloatTensor())
    state_dict['next_state'] = map_gpu(torch.FloatTensor())
    state_dict['timestep'] = map_gpu(torch.LongTensor())
    state_dict['final'] = map_gpu(torch.FloatTensor())
    n_steps = args.S
    n_critic = 5
    n_generator = 10
    # net.eval()
    # net.train()

    x_prev_multiplier, theta_multiplier, std, diffusion_steps_list = VAR_get_params(size, diffusion_hyperparams, user_defined_eta, kappa, continuous_steps)

    def update_f_v(x_seq, img, state_dict):

        x0 = x_seq[-1]
        f.train()
        v.train()
        # take the last s_0 to compute f; then update v(s_0) - v(s_T)
        # f = fstar()
        # optimizer_fstar = optim.RMSprop(f.parameters(), lr=1e-3)
        output = f(rescale_train(torch.cat((img.detach(),x0.detach()),0)))
            
        d_loss = output[:x0.shape[0]].mean()-output[x0.shape[0]:].mean()
        # print('mean check0', output[:x0.shape[0]].mean())
        # print('mean check1', output[x0.shape[0]:].mean())
        # print('d loss', d_loss)
        
        d_loss.backward()

        
        permutation = torch.randperm(args.batchsize * n_steps)
        # for count in range(0, n_steps*args.batchsize, n_steps*args.batchsize):
            # newest data
            # print(state_dict['state'].shape[0])
            # print(args.batchsize)
        indices = permutation + (state_dict['state'].shape[0] - (args.batchsize * n_steps))
        v_loss = F.mse_loss(v(state_dict['state'][indices], state_dict['timestep'][indices]), f(rescale_train(state_dict['final'][indices])))*n_steps
        v_loss.backward()
        # if args.local_rank == 0: print(v_loss/10)
        optimizer_v.step()
        optimizer_v.zero_grad()
        optimizer_fstar.step()
        optimizer_fstar.zero_grad()
        
        
        return f(rescale_train(torch.cat((img.detach(),x0.detach())),0))[x0.shape[0]:]

    for step, (images, labels) in enumerate(dataloader):
        assert (images.max().item() <= 1) and (0 <= images.min().item())
        # net.train()

        x_seq, log_prob_list = VAR_sampling(net, size,
                            diffusion_hyperparams,
                            user_defined_eta,
                            kappa=generation_param["kappa"],
                            continuous_steps=continuous_steps)
        log_prob_tensor = torch.cat(log_prob_list)
        
        for t in range(n_steps):
            state_dict['state'] = torch.cat((state_dict['state'],x_seq[t].detach()))
            state_dict['next_state'] = torch.cat((state_dict['next_state'],x_seq[t+1].detach()))
            state_dict['timestep'] = torch.cat((state_dict['timestep'], map_gpu(torch.tensor([t]*args.batchsize))))
            state_dict['final']= torch.cat((state_dict['final'],x_seq[-1].detach()))
        
        # test
        # pred_recalc = VAR_log_prob(net, state_dict['state'][:4], state_dict['next_state'][:4],state_dict['timestep'][:4], x_prev_multiplier, theta_multiplier, std, diffusion_steps_list, x_seq, log_prob_tensor)
            
        images  = 2 * map_gpu(images)- 1
        images = map_gpu(images)

        # test

        f_adv = update_f_v(x_seq, images, state_dict)
        # test_adv = f(rescale_train(state_dict['final']))
        norm = torch.tensor(0.0)

        if (step+1)%n_critic == 0:
            permutation = torch.randperm(state_dict['state'].shape[0])
            for m in range(0, args.batchsize*n_generator, args.batchsize):
                optimizer.zero_grad()
                indices = permutation[m:m + args.batchsize]
                with torch.no_grad():
                    adv = (f(rescale_train(state_dict['final'][indices]))-v(state_dict['state'][indices],state_dict['timestep'][indices])).detach().squeeze()
                if train == True:
                    with torch.enable_grad():
                        log_prob_t = VAR_log_prob(net, state_dict['state'][indices], state_dict['next_state'][indices],state_dict['timestep'][indices], x_prev_multiplier, theta_multiplier, std, diffusion_steps_list, x_seq, log_prob_tensor)

                        loss = (adv * log_prob_t).mean()
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                        optimizer.step()
                        # ema.update()

            state_dict = {}
            state_dict['state'] = map_gpu(torch.FloatTensor())
            state_dict['next_state'] = map_gpu(torch.FloatTensor())
            state_dict['timestep'] = map_gpu(torch.LongTensor())
            state_dict['final'] = map_gpu(torch.FloatTensor())
        
        if (step+1) % 10 == 0:
            if args.local_rank == 0:
                print(step)
                with torch.no_grad():
                    output = f(rescale_train(torch.cat((x_seq[-1].detach(),images.detach()),0)))
                    val_loss = output[:x_seq[-1].shape[0]].mean()-output[x_seq[-1].shape[0]:].mean()
                    print("val",val_loss.item())
                    print('norm', norm.item())
                    print('mean', output[:x_seq[-1].shape[0]].mean().item())
        if (step+1) % 200 == 0:
            save_image(make_grid(rescale(x_seq[-1])[:64]), fp=os.path.join('generated', '{}5-5-rms-noscale64-test.jpg'.format(output_name)))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument('-name', '--name', type=str, default = 'cifar10', choices=["cifar10", "lsun_bedroom", "lsun_church", "lsun_cat", "celeba64"],
                        help='Name of experiment')
    parser.add_argument('-n_channels', '--n_channels', type = int, default = '3')
    parser.add_argument('-img_shape', '--img_shape', type = int, default = '32')
    parser.add_argument('-ema', '--ema', help='Whether use ema', default = True)

    # fast generation parameters
    parser.add_argument('-approxdiff', '--approxdiff', type=str, default = 'VAR', choices=['STD', 'STEP', 'VAR'], help='approximate diffusion process')
    parser.add_argument('-kappa', '--kappa', type=float, default=1.0, help='factor to be multiplied to sigma')
    parser.add_argument('-S', '--S', type=int, default=10, help='number of steps')
    parser.add_argument('-schedule', '--schedule', type=str, choices=['linear', 'quadratic'], default = 'quadratic', help='noise level schedules')

    # generation util
    # parser.add_argument('-n', '--n_generate', type=int, help='Number of samples to generate', default = 50048)
    parser.add_argument('-dataset-path', '--dataset-path', type=str, default = './data')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs', default = 200)
    parser.add_argument('-bs', '--batchsize', type=int, default=128, help='Batchsize of generation')
    parser.add_argument('-gpu', '--gpu', type=str, default='cuda', choices=['cuda']+[str(i) for i in range(16)], help='gpu device')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # print(args.local_rank)
    args.gpu = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    global map_gpu
    map_gpu = _map_gpu(args.gpu)

    from config import model_config_map
    model_config = model_config_map[args.name]
    
    kappa = args.kappa
    if args.approxdiff == 'VAR':  # user defined variance
        user_defined_eta = get_VAR_noise(args.S, args.schedule)
        generation_param = {"kappa": kappa, 
                            "user_defined_eta": user_defined_eta}
        variance_schedule = '{}{}'.format(args.S, args.schedule)

    else:
        raise NotImplementedError

    output_name = '{}{}_{}{}_kappa{}'.format('ema_' if args.ema else '',
                                             args.name, 
                                             args.approxdiff,
                                             variance_schedule,
                                             kappa)

    # model_path = os.path.join('./ema_celeba64_VAR10quadratic_kappa1.0finetuned_test.pt')
    model_path = os.path.join('checkpoints', 
                                '{}diffusion_{}_model'.format('ema_' if args.ema else '', args.name), 
                                'model.ckpt.pth')
    # map diffusion hyperparameters to gpu
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = map_gpu(diffusion_hyperparams[key])

    # predefine model
    net = Model(**model_config)
    print_size(net)
    f = map_gpu(Discriminator())
    v = map_gpu(Value(num_steps = args.S, img_shape = (args.n_channels,args.img_shape, args.img_shape)))
    # load checkpoint
    try:
        d = fix_legacy_dict(torch.load(model_path, map_location='cpu'))
        dm = net.state_dict()
        # for k in args.delete_keys:
        #     print(
        #         f"Deleting key {k} because its shape in ckpt ({d[k].shape}) doesn't match "
        #         + f"with shape in model ({dm[k].shape})"
        #     )
        #     del d[k]
        net.load_state_dict(d, strict=False)
        # checkpoint = torch.load(model_path, map_location='cpu')
        # net.load_state_dict(checkpoint)
        net = map_gpu(net)
        net.eval()
        # net.train()
        print('checkpoint successfully loaded')
    except:
        raise Exception('No valid model found')

    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        if args.local_rank == 0:
            print(f"Using distributed training on {ngpus} gpus.")
        args.batchsize = args.batchsize // ngpus
        # print("actual batchsize:", args.batch_size)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)
        # add or not?
        f = DDP(f, device_ids=[args.local_rank], output_device=args.local_rank)
        v = DDP(v, device_ids=[args.local_rank], output_device=args.local_rank)


    ema = ExponentialMovingAverage(net.parameters(), decay=0.995)
    # diffusion params
    user_defined_eta = generation_param["user_defined_eta"]
    continuous_steps = _precompute_VAR_steps(diffusion_hyperparams, user_defined_eta)
    C, H, W = model_config["in_channels"], model_config["resolution"], model_config["resolution"]

    # train loader
    # metadata = get_metadata(args.name)
    train_set = get_dataset(args.name, args.dataset_path)
    sampler = DistributedSampler(train_set) if ngpus > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batchsize,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)
    optimizer_fstar = torch.optim.Adam(f.parameters(), lr=1e-4)
    optimizer_v = torch.optim.Adam(v.parameters(), lr=1e-4)   

    # train
    for epoch in range(args.n_epochs):
        if args.local_rank == 0:
            if epoch > 0:
                # # with ema.average_parameters():
                Xi, log_prob_list = VAR_sampling(net, (64, C, H, W),
                                diffusion_hyperparams,
                                user_defined_eta,
                                kappa=generation_param["kappa"],
                                continuous_steps=continuous_steps)
                Xi = Xi[-1]
            
                # save image

                save_image(make_grid(rescale(Xi)[:64]), fp=os.path.join('generated', '{}.jpg'.format(output_name)))
                torch.save(net.state_dict(), '{}finetuned_test.pt'.format(output_name))
                # with ema.average_parameters():
                    # torch.save(net.state_dict(), '{}finetuned_0.5-5-5-adam-ema.pt'.format(output_name))
            else:
                Xi, log_prob_list = VAR_sampling(net, (64, C, H, W),
                                diffusion_hyperparams,
                                user_defined_eta,
                                kappa=generation_param["kappa"],
                                continuous_steps=continuous_steps)
                Xi = Xi[-1]
            
                # # # # # save image
                save_image(make_grid(rescale(Xi)[:64]), fp=os.path.join('generated', '{}init.jpg'.format(output_name)))
                # # # torch.save(net.state_dict(), 'finetuned_ema_1.0.pt')
            print('epoch', epoch)

        train_one_epoch(net=net, ema = ema, dataloader = train_loader, optimizer = optimizer, f = f, v = v, optimizer_fstar = optimizer_fstar, optimizer_v = optimizer_v, continuous_steps = continuous_steps, diffusion_hyperparams = diffusion_hyperparams, size = (args.batchsize, C, H, W), user_defined_eta=user_defined_eta, kappa = kappa, args = args, train = (epoch >-1))


