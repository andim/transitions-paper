import os.path, sys
sys.path.append('../lib/')
import numpy as np
import evolimmune
from evolimmune import to_api, from_api, from_tau
from misc import *
import noisyopt
import scipydirect

# model definitions
f1 = np.array([1.0, 0.3]) 
f2 = np.array([0.4, 1.0])
# parameters of numerical algorithms
niter = 1e6
nburnin = 1e4
# first stage numerical parameters
maxfs = 100
# second stage numerical parameters
deltainit = 0.02
deltatols = 0.005
significancealpha = 0.005
feps = 1e-6
# script parameters
# hq scan
aenvs = from_tau(np.logspace(np.log10(0.09), np.log10(20.0), 40, True))
pienvs = np.linspace(0.0, 1.0, 41)#[1:-1]
# lq scan
#aenvs = from_tau(np.logspace(np.log10(0.09), np.log10(20.0), 20, True))
#pienvs = np.linspace(0.0, 1.0, 21)#[1:-1]

nbatch = 10
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

        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        fargs = f1, f2, aenv, pienv, niter, nburnin
        def obj(x, *args, **kwargs):
            return -evolimmune.Lambda_f(evolimmune.switching_matrix_2by2(x), *args, **kwargs)
        res1 = scipydirect.minimize(obj, maxf=maxf, args=fargs, bounds=bounds, disp=disp)
        if disp:
            print 'results of first phase optimization', res1
        res2 = noisyopt.minimize(obj, res1.x,
                   args=fargs, bounds=bounds,
                   deltainit=deltainit, deltatol=deltatol,
                   alpha=significancealpha, feps=feps,
                   errorcontrol=True, paired=True,
                   disp=disp)
        #res2.x[res2.free] = np.nan
        alpha, beta = res2.x
        if disp:
            print 'result', res2.x
        Lambdaopt = res2.fun
        Lambdaoptse = res2.funse
        data.append([f1[0], f1[1], f2[0], f2[1], aenv, pienv,
            Lambdaopt, Lambdaoptse, alpha, beta,
            niter, nburnin, maxf, deltainit, deltatol, significancealpha, np.log10(feps)])
    columns = ['f11', 'f21', 'f12', 'f22', 'aenv', 'pienv', 'Lambdaopt', 'Lambdaoptse',
               'alpha', 'beta',
               'niter', 'nburnin', 'maxf', 'deltainit', 'deltatol', 'significancealpha', 'logfeps',
               ]
    np.savez_compressed(datadir + 'scan_opt%g' % (njob), data=data, columns=columns)
