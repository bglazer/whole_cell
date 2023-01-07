import torch
from functools import partial
import random

def f_pow(exp):
    f = partial(torch.pow, exponent=torch.tensor(exp))
    f.__name__ = f'pow_{exp}'
    return f

def random_function():
    library = [torch.sin, torch.cos, f_pow(2), f_pow(3)]
    n = random.randint(1, len(library)-1)
    terms = random.sample(library, k=n)
    ks = [random.uniform(-1, 1) for _ in range(n)]
    return lambda x: sum([ks[i]*terms[i](x) for i in range(n)]), terms, ks

if __name__ == '__main__':
    f, ts, ks = random_function()
    x = torch.linspace(-1, 1, 10)
    print(f(x))
    print('+'.join([f'{k:.2f}*{t.__name__}(x)' for k, t in zip(ks, ts)]))

