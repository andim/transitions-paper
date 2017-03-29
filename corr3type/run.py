import os.path, sys
sys.path.append('../lib/')
import numpy as np
import evolimmune
from evolimmune import to_api, from_api, from_tau
from misc import *
import noisyopt
import scipydirect

# model definitions
f1 = np.array([1.0, 0.8, 0.3]) 
f2 = np.array([0.2, 0.7, 1.0])
# parameters of numerical algorithms
niter = 1e6
nburnin = 1e4
# first stage numerical parameters
maxfs = 10000
# second stage numerical parameters
deltainit = 0.02
deltatols = 0.01
significancealpha = 0.05
feps = 1e-6
# script parameters
aenvs = from_tau(np.logspace(np.log10(0.09), np.log10(20.0), 40, True))
pienvs = np.linspace(0.0, 1.0, 41)

nbatch = 5
disp = True
datadir = 'data/'

paramscomb = params_combination((aenvs, pienvs, maxfs, deltatols))
niter = int(niter)
nburnin = int(nburnin)
if parametercheck(datadir, sys.argv, paramscomb, nbatch):
    njob = int(sys.argv[1])
    data = []
    for i in progressbar(range(nbatch)):
        n = (njob-1) * nbatch + i
        aenv, pienv, maxf, deltatol = paramscomb[n]
        if disp:
            print paramscomb[n]

        bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        fargs = f1, f2, aenv, pienv, niter, nburnin
        def obj(x, *args, **kwargs):
            return -evolimmune.Lambda_f(evolimmune.switching_matrix_3by3(x), *args, **kwargs)
        res1 = scipydirect.minimize(obj, maxf=maxf, args=fargs, bounds=bounds, disp=disp)
        if disp:
            print 'results of first phase optimization', res1
        res2 = noisyopt.minimize(minus(evolimmune.Lambda_f3), res1.x,
                   args=fargs, bounds=bounds,
                   deltainit=deltainit, deltatol=deltatol,
                   alpha=significancealpha, feps=feps,
                   errorcontrol=True, paired=True,
                   disp=disp)
        alpha, beta, gamma, delta, epsilon, zeta = res2.x
        if disp:
            print 'result', res2.x
        Lambdaopt = res2.fun
        Lambdaoptse = res2.funse
        data.append([f1[0], f1[1], f1[2], f2[0], f2[1], f2[2], aenv, pienv,
            Lambdaopt, Lambdaoptse, alpha, beta, gamma, delta, epsilon, zeta,
            niter, nburnin, maxf, deltainit, deltatol, significancealpha, np.log10(feps)])
    columns = ['f11', 'f21', 'f31', 'f12', 'f22', 'f32', 'aenv', 'pienv', 'Lambdaopt', 'Lambdaoptse',
               'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
               'niter', 'nburnin', 'maxf', 'deltainit', 'deltatol', 'significancealpha', 'logfeps',
               ]
    np.savez_compressed(datadir + 'scan_opt%g' % (njob), data=data, columns=columns)
