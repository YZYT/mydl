from collections import defaultdict
import torch.nn as nn
import torch
from torch.optim.optimizer import Optimizer
import random

class BasePSSGD(Optimizer):    
    def __init__(self):
        raise NotImplementedError
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)


    @property
    def param_groups(self):
        return self.optimizer.param_groups


class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.5, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss






class PSSGDv1(Optimizer):
    r"""Version 1.0
    We update the fast parameter la_steps times, and then use the average as the update of slow parameter.
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                for i in range(la_steps):
                    param_state['avg'] = torch.zeros_like(p.data)
                    param_state['last_' + str(i)] = torch.zeros_like(p.data)
                    param_state['last_' + str(i)].copy_(p.data)


    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['last_' + str(self._la_step)].copy_(p.data)
                param_state['avg'].add_(p.data, alpha=1/self._total_la_steps)
                # print(f"{self._la_step}: {param_state['avg']}")

        
        loss = self.optimizer.step(closure)
        self._la_step += 1
        
        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.copy_(param_state['avg'])
                    param_state['avg'].zero_()

        return loss

class PSSGDv2(Optimizer):
    r"""Version 2.0
    We update the fast parameter la_steps times, and then use the average of the first and last parameter as the update of slow parameter. 
    The same as Lookahead.
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.5, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                for i in range(la_steps):
                    param_state['avg'] = torch.zeros_like(p.data)
                    param_state['last_' + str(i)] = torch.zeros_like(p.data)
                    param_state['last_' + str(i)].copy_(p.data)


    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['last_' + str(self._la_step)].copy_(p.data)


        loss = self.optimizer.step(closure)
        self._la_step += 1
        
        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['last_0'], alpha=1.0 - self.la_alpha)  # crucial line
        return loss


class PSSGDv3(Optimizer):
    r"""Version 3.0
    Randomly choose a past point
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.5, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                for i in range(la_steps):
                    param_state['avg'] = torch.zeros_like(p.data)
                    param_state['last_' + str(i)] = torch.zeros_like(p.data)
                    param_state['last_' + str(i)].copy_(p.data)


    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['last_' + str(self._la_step)].copy_(p.data)


        loss = self.optimizer.step(closure)
        self._la_step += 1
        
        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # randomly select a point 
            k = random.randint(0, self._total_la_steps - 1)
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['last_'+str(k)], alpha=1.0 - self.la_alpha)  # crucial line
        return loss


class PSSGDv4(BasePSSGD):
    r"""Version 4.0
    multiple points
    """

    def __init__(self, optimizer, loss_fn=None, k=5, la_steps=5, la_alpha=0.5, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        if loss_fn == None:
            loss_fn = nn.MSELoss(reduction='mean')
        self.loss_fn = loss_fn
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps

        # number
        self.k = k

        # PSO state
        self.age = 0
        self.steps = 0
        self.id = k
        
        self.state = [defaultdict(dict) for _ in range(k + 1)]
        self.fitness = [-1 for _ in range(k + 1)]

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for i in range(k + 1):
                for p in group['params']:
                    param_state = self.state[i][p]
                    # param_state['avg'] = torch.zeros_like(p.data)
                    # param_state['last_' + str(i)] = torch.zeros_like(p.data)
                    # param_state['last_' + str(i)].copy_(p.data)
                    param_state['weight'] = torch.zeros_like(p.data)
                    param_state['weight'].copy_(p.data)
                    param_state['age'] = 0   
                    param_state['steps'] = 0        


    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'k': self.k,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
        }

    # def zero_grad(self):
    #     self.optimizer.zero_grad()

    # def get_la_step(self):
    #     return self._la_step

    # def state_dict(self):
    #     return self.optimizer.state_dict()

    # def load_state_dict(self, state_dict):
    #     self.optimizer.load_state_dict(state_dict)


    # @property
    # def param_groups(self):
    #     return self.optimizer.param_groups

    def select_victim(self):
        victim = 0
        for i in range(self.k + 1):
            if self.fitness[i] < 0:
                return i
            if self.fitness[i] > self.fitness[victim]:
                victim = i
        print(self.fitness)
        return victim

    def setloss(self, l):
        victim = self.select_victim()
        self.store_state(victim)
        self.fitness[victim] = l

    def select_father(self):
        return 0

    def _select_candicate(self):
        return self.k
    
    def select_candicate(self):
        # select an individual to grow
        candicate = self._select_candicate()
        self.load_state(candicate)


    def store_state(self, k):
        """
        store the current position into kth buffer.
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[k][p]
                param_state['weight'].copy_(p.data)
                param_state['age'] = self.age
                param_state['steps'] = self.steps
    
    def load_state(self, k):
        self.id = k
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[k][p]
                p.data.copy_(param_state['weight'])
                self.steps = param_state['steps']
                self.age = param_state['age']



    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """


        # ======= backup the current state =======



        # ======= SGD step =======
        # SGD step (individual grow)
        loss = self.optimizer.step(closure)
        # print(f"loss: {loss}")
        self._la_step += 1

        self.age += 1
        self.steps += 1

        # for group in self.optimizer.param_groups:
        #     for p in group['params']:
        #         print(p.data)

        # ======= evolution stage =======
        if self._la_step >= self._total_la_steps * 10000:
            self._la_step = 0

            # find a victim
            victim = self.select_victim()

            # select an individual
            father = self.select_father()

            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state_new = self.state[victim][p]
                    param_state_fa = self.state[father][p]

                    param_state_new['weight'].copy_(p.data.mul(self.la_alpha).add(param_state_fa['weight'], alpha=1.0-self.la_alpha))

                    param_state_new['age'] = (self.age + param_state_fa['age']) / 2
                    param_state_new['steps'] = 0 
                    
                    print(f"victim: {victim}")
                    print(f"father: {father}")
                    print(f"p.data: {p.data}")
                    print(f"father: {param_state_fa['weight']}")
                    print(f"new: {param_state_new['weight']}")

                    # print("using")
                    # print(self.state)
        
        return loss

# optimizer = # {any optimizer} e.g. torch.optim.Adam
# if args.lookahead:
#     optimizer = Lookahead(optimizer, la_steps=args.la_steps, la_alpha=args.la_alpha)