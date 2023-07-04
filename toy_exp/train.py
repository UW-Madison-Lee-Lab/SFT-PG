import argparse
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import ot
from models import ConditionalModel, value, fstar_tanh

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
# import torch.multiprocessing as mp

# from mlp import ConditionalModel, Classifier, fstar, gradient_penalty
from sklearn.datasets import make_swiss_roll
from torch.distributions import Normal

DEVICE = 'cpu'

from sklearn import datasets
n_samples = 10000
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state = 0)
#
train_set, train_set_y = noisy_moons
train_set = torch.tensor(train_set).float()
train_set = train_set/2
train_set_y = torch.tensor(train_set_y).long().reshape((train_set_y.shape[0],))

# swiss roll
train_set, _ = make_swiss_roll(n_samples, noise=0.05)
train_set = train_set[:, [0,2]]/20
train_set = torch.tensor(train_set).float()

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def noise_estimation_loss(model, x_0):
    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, x_0).to(DEVICE)
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, x_0).to(DEVICE)
    e = torch.randn_like(x_0)
    # model input
    x = x_0 * a + e * am1
    x = x.to(DEVICE)
    t = t.to(DEVICE)
    output = model(x, t)
    return (e - output).square().mean()


def p_sample(model, x, t):
    t = torch.tensor([t])
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_bar_sqrt, t, x))
    # Model output
    eps_theta = model(x.clone().to(DEVICE), t.clone().to(DEVICE))
    eps_theta = eps_theta.cpu()
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x)
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return (sample)

def p_sample_loop(model, shape):
    cur_x = torch.randn(shape)

    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i)
        x_seq.append(cur_x)
    return x_seq

def p_sample_finetune_dist(model, x, t, x_next):
    # t = torch.tensor([t])
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_bar_sqrt, t, x))
    # Model output
    eps_theta = model(x.clone().to(DEVICE), t.clone().to(DEVICE))
    # eps_theta = eps_theta.cpu()
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x)
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    # sample = mean + sigma_t * z
    dist = Normal(mean, sigma_t)

    return dist.log_prob(x_next.detach()).sum(dim = -1)




class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


if __name__ == '__main__':
    # add argument parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--finetune_lr', type=float, default=5e-5)
    parser.add_argument('--finetune_f_lr', type=float, default=1e-3)
    parser.add_argument('--finetune_v_lr', type=float, default=1e-3)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--n_generator', type=int, default=1)
    args = parser.parse_args()


    n_steps = args.n_steps
    betas = make_beta_schedule(schedule='sigmoid', n_timesteps=n_steps, start=1e-5, end=1e-2)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    model = ConditionalModel(n_steps)
    model.to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=args.pretrain_lr)
    # dataset = train_set.data.clone().detach().float()
    # Create EMA model
    ema = EMA(0.9)
    ema.register(model)
    # Batch size
    batch_size = args.batch_size
    train_gen = False
    train_class = False
    train_train_with_gen = True
    test_gen_with_class = False
    dataset = train_set
    # dataset = train_set[train_set_y == 0]

    if args.pretrain:
        for t in range(args.n_epochs):
            # X is a torch Variable
            permutation = torch.randperm(dataset.size()[0])
            for i in range(0, dataset.size()[0], batch_size):
                # Retrieve current batch
                indices = permutation[i:i + batch_size]
                batch_x = dataset[indices].to(DEVICE)
                # plt.scatter(batch_x[:, 0], batch_x[:, 1], s=10)
                # axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], s=10);
                # Compute the loss.
                loss = noise_estimation_loss(model, batch_x)
                # Before the backward pass, zero all of the network gradients
                optimizer.zero_grad()
                # Backward pass: compute gradient of the loss with respect to parameters
                loss.backward()
                # Perform gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                # Calling the step function to update the parameters
                optimizer.step()
                # Update the exponential moving average
                ema.update(model)
            # Print loss
            if t % 100 == 0:
                print(loss)
                x_seq = p_sample_loop(model, dataset.shape)
                plt.scatter(x_seq[-1][:, 0].detach(), x_seq[-1][:, 1].detach(), s=5);
                plt.show()
        torch.save(model.state_dict(), "swiss.pt")

    else:
        model.load_state_dict(torch.load("swiss.pt"))
        model.to(DEVICE)
        model.train()
        # classifier = classifier_new(n_steps)
        # classifier.load_state_dict(torch.load("points_imbalancedclassifier.pt"))
        optimizer_class = optim.Adam(model.parameters(), lr=args.finetune_lr)
        f = fstar_tanh().to(DEVICE)
        optimizer_fstar = optim.Adam(f.parameters(), lr=args.finetune_f_lr)
        v = value(n_steps).to(DEVICE)
        optimizer_v = optim.Adam(v.parameters(), lr=args.finetune_v_lr)
        state_dict = []

        def update_f_v(x_seq, x1):
            optimizer_fstar.zero_grad() 
            optimizer_v.zero_grad()
            x0 = x_seq[-1]
            output = f(torch.cat((x1.detach(),x0.detach()),0))
                
            d_loss = output[:x0.shape[0]].mean()-output[x0.shape[0]:].mean() 
            d_loss.backward()

            v_loss = 0
            for t in range(n_steps):
                v_loss += F.mse_loss(v(x_seq[t], torch.tensor(t).to(DEVICE)), f(x0))/n_steps
            v_loss.backward()
            optimizer_fstar.step()           
            optimizer_v.step()
            

        state_dict = {}
        state_dict['state'] = torch.FloatTensor().to(DEVICE)
        state_dict['next_state'] = torch.FloatTensor().to(DEVICE)
        state_dict['timestep'] = torch.IntTensor().to(DEVICE)
        state_dict['final'] = torch.FloatTensor().to(DEVICE)
        state_dict['valuestep'] = torch.IntTensor().to(DEVICE)

        for j in range(300):
            train_permutation = torch.randperm(train_set.size()[0])
            k = 0
            for i in range(0, train_set.size()[0], batch_size):
                k+=1
                model.eval()
                # x_seq, log_prob = p_sample_loop_finetune(model, (batch_size, 2))
                x_seq = p_sample_loop(model, (batch_size, 2))
                for t in range(n_steps):
                    state_dict['state'] = torch.cat((state_dict['state'],x_seq[t].detach()))
                    state_dict['next_state'] = torch.cat((state_dict['next_state'],x_seq[t+1].detach()))
                    state_dict['timestep'] = torch.cat((state_dict['timestep'], torch.tensor([n_steps-(t+1)]*batch_size).to(DEVICE)))
                    state_dict['valuestep'] = torch.cat((state_dict['valuestep'], torch.tensor([t]*batch_size).to(DEVICE)))
                    state_dict['final']= torch.cat((state_dict['final'],x_seq[-1]))

                train_indices = train_permutation[i:i+batch_size]
                train_x = train_set[train_indices].to(DEVICE)
                update_f_v(x_seq, train_x)

                if (k+1)% args.n_critic == 0:
                    # f.eval()
                    model.train()
                    permutation = torch.randperm(state_dict['state'].shape[0])
                    for m in range(0, batch_size, batch_size*args.n_generator):
                        optimizer_class.zero_grad()
                        indices = permutation[m:m + batch_size]
                        log_prob_t = p_sample_finetune_dist(model, state_dict['state'][indices], state_dict['timestep'][indices], state_dict['next_state'][indices])
                        adv = (f(state_dict['final'][indices])-v(state_dict['state'][indices],state_dict['valuestep'][indices])).detach().squeeze()
                        loss = (adv * log_prob_t).mean()
                        loss.backward()
                        optimizer_class.step()


                    state_dict['state'] = torch.FloatTensor().to(DEVICE)
                    state_dict['next_state'] = torch.FloatTensor().to(DEVICE)
                    state_dict['timestep'] = torch.IntTensor().to(DEVICE)
                    state_dict['final'] = torch.FloatTensor().to(DEVICE)
                    state_dict['valuestep'] = torch.IntTensor().to(DEVICE)

            if (j)% 10 == 0:
                # plt.figure(figsize=(8,8))
                plt.xlim(-1.1, 1.1)
                plt.ylim(-1.1, 1.1)
                # plt.show()
                x_seq = p_sample_loop(model, train_set.shape)

                plt.scatter(x_seq[-1][:, 0].detach().cpu(), x_seq[-1][:, 1].detach().clone().cpu(), c = 'blue',s=1)
                plt.scatter(train_set[:,0], train_set[:, 1], c = 'red', s = 1, alpha = 0.1)
                # plt.savefig("model_epoch_{}_swiss55-norm.png".format(j))
                plt.close()
                
                # plt.title(label="Prob of class 0")
                print(j)
                val_loss = f(x_seq[-1]).detach().mean()-f(train_set.to(DEVICE)).detach().mean()
                print("val",val_loss)
                # scheduler.step(val_loss)
                # plt.show()
                w_dis = ot.sliced_wasserstein_distance(train_set, x_seq[-1].detach().cpu(), n_projections=100, p = 2)
                print("w_dis", w_dis)

        torch.save(model.state_dict(), "swiss_finetuned.pt")
