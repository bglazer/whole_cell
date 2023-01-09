#%%
import torch
from functools import partial
import random

def f_pow(exp):
    f = partial(torch.pow, exponent=torch.tensor(exp))
    f.__name__ = f'pow_{exp}'
    return f

library = [torch.sin, torch.cos]#, f_pow(1)] #, f_pow(3)]

def random_function(library, krange):
    term = random.choice(library)
    k = random.uniform(-krange, krange)
    return term, k

def random_system(n_variables, library, krange, max_terms, self_deg=True):
    terms = []
    system = []
    for i in range(n_variables):
        n_inputs = random.randint(1, min(max_terms, n_variables))
        input_choices = random.sample(range(n_variables), k=n_inputs)
        eqn = [None for _ in range(n_variables)]
        fn = [None for _ in range(n_variables)]
        for input_choice in input_choices:
            term, k = random_function(library, krange)
            eqn[input_choice] = (k, term)
            fn[input_choice] = lambda x,k=k,term=term: k*term(x)
        if self_deg:
            k = random.uniform(0, krange)
            fn[i] = lambda x,k=k: -k*x
        # Create a lambda function that sums the terms
        system.append(lambda xs, fn=fn: sum([f(x) for f,x in zip(fn, xs) if f is not None]))
        terms.append(eqn)
    system = lambda xs, s=system: torch.stack([f(xs) for f in s])
    return system, terms

def print_system(system):
    for i,eqn in enumerate(system):
        print(f'x_{i} =', '+'.join([(f'{t[0]:.2f}*{t[1].__name__}(x_{j})') 
              for j,t in enumerate(eqn) if t is not None]))

#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    system, terms = random_system(3, library, 1/3, 3)
    print_system(terms)
    xs = torch.randn(3)
    # print(system)
    h = .01
    # Run Euler's method for 10 steps
    n = 1000
    us = torch.zeros(n, 3)
    for j in range(n):
        # Euler step
        u = system(xs)
        #print(', '.join(f'{x.item(): .3f}' for x in u))
        xs = xs + h*u
        us[j] = xs
    plt.plot(us[:,0])
    plt.plot(us[:,1])
    plt.plot(us[:,2])
    plt.show()

# %%
