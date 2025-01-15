#Version of prep_limit_new where the data is regularised in both the x and y scale
import ROOT
ROOT.EnableImplicitMT()
import os, sys, glob, pickle, argparse, subprocess, multiprocessing, itertools
# import jax
# import jax.numpy as jnp
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
from iminuit import Minuit
from iminuit import minimize as MinuitMinimize
from functools import partial
from scipy.stats import chi2, norm
import random
# plt.style.use(hep.style.ROOT)
cols = ['#4285f4','#ea4335','#fbbc05','#34a853', '#a00498', '#536267']

def exponentiated_quadratic(xa, xb,variance,scale):
    """Exponentiated quadratic  with K=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')/scale**2
    return variance*np.exp(sq_norm)

# @jax.jit
def GP_noise(params,X1, y1, X2, kernel_func,noise):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the noisy observations 
    (y1, X1), and the prior kernel function.

    Used for inference once the kernel hyperparameters are determined
    """
    K11 = kernel_func(X1, X1,np.exp(params[0]),np.exp(params[1])) + np.diag(noise**2)
    K12 = kernel_func(X1, X2,np.exp(params[0]),np.exp(params[1]))
    solved = scipy.linalg.solve(K11, K12, assume_a='pos').T
    μ2 = solved @ y1
    K22 = kernel_func(X2, X2,np.exp(params[0]),np.exp(params[1]))
    K2 = K22 - (solved @ K12)
    return μ2, np.sqrt(np.diag(K2))

def gauss_array(x, mu=100, sig=10, bw=1, a=1):
#     print(x.shape,mu,sig,bw)
    """
    Simple Gaussian array (used for signal)
    """
    y = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    #den = sig * ((2*np.pi)**0.5)
    den = 1
    y=y*bw*a/den
    # assert np.around(np.sum(y)*(x[1]-x[0]))==bw, (np.sum(y), "is not 1", mu, sig, bw)
    return y

# Define the likelihood function
def gp_lml(params, X, y, noise, kernel_func):
    """
    Components:
    * X,y: From input histogram
    * K11: Kernel created from x values of the bins
        Hyperparameters:
        - length_scale: Horizontal scale
        - variance: Vertical scale
    * Noise: Noise term for each bin (usually sqrt(N))
    * kernel_func: Kernel function (like RBF)
    """
    K11 = kernel_func(X, X,np.exp(params[0]),np.exp(params[1])) + np.diag(noise**2)
    _, K11_logdet = np.linalg.slogdet(K11)
    return 0.5*(y.T @ np.linalg.inv(K11) @ y) + 0.5*K11_logdet + 0.5*np.float64(y.shape[0])*np.log(2*np.pi)
    #return 0.5*(y.T @ np.linalg.inv(K11) @ y) + 0.5*np.log(np.linalg.det(K11)) + 0.5*np.float64(y.shape[0])*np.log(2*np.pi)

# Likelihood function for signal hypothesis
def gp_sig_lml(params,x, y, noise):
    """
    """
    # print(params)
    K11 = exponentiated_quadratic(x[:,None], x[:,None],np.exp(params[0]),np.exp(params[1])) + np.diag(noise**2)
    sig = params[2]*gauss_array(x,params[3],params[4],x[1]-x[0])
    Y = (y - sig)[:,None]
    lml=0.5*(Y.T @ np.linalg.inv(K11) @ Y) + 0.5*np.log(np.linalg.det(K11)) + 0.5*np.float64(Y.shape[0])*np.log(2*np.pi)
    return lml.flatten()[0]

def sample_gp(params,X1, y1, X2, kernel_func,noise,ny):
    K11 = kernel_func(X1, X1,np.exp(params[0]),np.exp(params[1])) + np.diag(noise**2)
    K12 = kernel_func(X1, X2,np.exp(params[0]),np.exp(params[1]))
    solved = scipy.linalg.solve(K11, K12, assume_a='pos').T
    μ2 = solved @ y1
    K22 = kernel_func(X2, X2,np.exp(params[0]),np.exp(params[1]))
    K2 = K22 - (solved @ K12)
    return np.random.multivariate_normal(mean=μ2.flatten(), cov=K2, size=ny)

def make_toy(Y_arr,hist,count):
    y_toys = np.zeros_like(Y_arr)
    for j,Y in enumerate(Y_arr):
        # Y = np.sum(Y_arr,axis=0)
        ROOT.gRandom.SetSeed(random.randint(0,9999))
        hb = hist.Clone('hb')
        hb.Reset()

        for i in range(1,h.GetNbinsX()+1):
            hb.SetBinContent(i,Y[i-1])

        htoy = hb.Clone('htoy')
        htoy.Reset()
        htoy.FillRandom(hb,count)

        y_toy=[]     
        for i in range(1,h.GetNbinsX()+1):
            if(h.GetBinContent(i)>-1):
                y_toy.append(htoy.GetBinContent(i))
        y_toys[j] = np.array(y_toy)

    return y_toys

def show_fit(x, y, dy, xc, yc, dyc, yp, sigma_p, mean, sigma, toys=None, amp=None, tag='fit', yscale='log'):
    """
    Variable definition
    Histogram variables:                x, y, dy
    Histogram variables in sig window:  xc, yc, dyc
    GP prediction:                      yp, sigma_p
    Signal parameters:                  mean, sigma
    List of y-counts from toys:         toys
    Signal strength:                    amp
    """
    x_ = np.linspace(x[0], x[-1], 1000)
    fig = plt.figure(figsize=(10, 13))
    spec = fig.add_gridspec(ncols=1, nrows=3,height_ratios=[0.8,0.2,0.3])
    main = fig.add_subplot(spec[0,0])
    if toys is not None:
        main.plot(x, toys[:10].T,'.',color='red',alpha=0.3)
    main.errorbar(x, y, dy, fmt='ok', label='Data')
    main.plot(x, yp, label='Prediction',color=cols[1])
    main.fill_between(x, yp-sigma_p, yp+sigma_p, label='68% CI',color=cols[1],alpha=0.2)
    main.axvline(x=mean-2*sigma,color='lightgrey',linestyle='--')
    main.axvline(x=mean+2*sigma,color='lightgrey',linestyle='--')
    main.plot([], [], ' ', label='{}/{}'.format(np.sum(np.square((y-yp)/np.sqrt(dy**2 + sigma_p**2))),y.shape[0]))
    if amp is not None:
        main.plot([], [], ' ', label='{}/{}'.format(np.sum(np.square((y-amp*gauss_array(x,mean,sigma,x[1]-x[0])-yp)/np.sqrt(dy**2 + sigma_p**2))),y.shape[0]))
    main.legend()
    if amp is not None:
        main.plot(x, amp*gauss_array(x,mean,sigma,x[1]-x[0]),label='Signal',color=cols[3])
    # Plotting scale
    ymax = np.max(y)
    ymin = np.min(y) 
    main.set_ylim(0.8*ymin, ymax*1.2)
    main.set_yscale(yscale)
    pull = fig.add_subplot(spec[1,0])
    pull.bar(x, (y-yp)/(np.sqrt(dy**2 + sigma_p**2)), width=x[1]-x[0], label='Data Pull',color=cols[1])
    if amp is not None:
        pull.bar(x, (y-amp*gauss_array(x,mean,sigma,x[1]-x[0])-yp)/(np.sqrt(dy**2)), width=x[1]-x[0], label='Signal Pull',color=cols[3])
    pull.axvline(x=mean-2*sigma,color='lightgrey',linestyle='--')
    pull.axvline(x=mean+2*sigma,color='lightgrey',linestyle='--')
    pull.set_ylim(-5,5)
    pull.legend()
    diff = fig.add_subplot(spec[2,0])
    diff.errorbar(x, y-yp, dy, fmt='ok', label='Data')
    diff.fill_between(x, -sigma_p, sigma_p, label='68% CI',color=cols[1],alpha=0.1)
    diff.axvline(x=mean-2*sigma,color='lightgrey',linestyle='--')
    diff.axvline(x=mean+2*sigma,color='lightgrey',linestyle='--')
    diff.axhline(y=0,color='lightgrey',linestyle='-')
    diff.set_ylim(np.min(y-yp-1.1*dy),np.max(y-yp+1.1*dy))
    if amp is not None:
        diff.plot(x_, amp*gauss_array(x_,mean,sigma,x[1]-x[0]),label='Signal',color=cols[3])
        min_ = np.min(y-yp-1.1*dy)
        max_ = max(np.max(y-yp+1.1*dy),np.max(amp*gauss_array(x_,mean,sigma,x[1]-x[0])))
        diff.set_ylim(min_,max_)
    plt.savefig(tag+str(int(mean))+'.pdf')

# params[3] and params[4] are the mean and sigma of the signal
def log_likelihood(params,x,y,dy,a):
    signal = params[2]*gauss_array(x,params[3],params[4],a=a)
    mask1 = x<(params[3]-2*params[4])
    mask2 = x>(params[3]+2*params[4])
    mask = np.logical_or(mask1,mask2)
    # print(mask)
    xc= x[mask]
    yc= y[mask]
    dyc= dy[mask]
    yp,_=GP_noise(params[:2],xc[:,None],yc[:,None],x[:,None],exponentiated_quadratic,dyc)
    # yp is returned in log scale so the signal is added in linear scale
    yp = yp.flatten()
    yp = np.exp(yp)
    y = np.exp(y)
    yp[yp<0]=1e-9
    Mu =  yp + signal + 1e-9
    yp_ = yp + 1e-9
    # print(xc.shape,yp.shape,Mu.shape)
    mask0 = y>0
    y = y[mask0]
    Mu = Mu[mask0]
    yp_ = yp_[mask0]
    Mu[Mu<=0] = 1e-3
    # Debug
    #print(x)
    #print(yp_)
    #print(Mu)
    # Convert al to linear before minimising
    ll = 2*np.sum(Mu-y+(y*np.log(y/Mu)))
    ll2 = 2*np.sum(yp_-y+(y*np.log(y/yp_)))
    # print(ll2-ll)
    return ll


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",default='resolved2016_reg2.root')
    parser.add_argument("--hist_name",default='h_mass')
    parser.add_argument("--length_scale",type=float,default=100)
    parser.add_argument("--variance",type=float,default=10)
    parser.add_argument("--mean",type=float,default=10)
    parser.add_argument("--sigma",type=float,default=10)
    parser.add_argument("--rate_uc",type=float,default=0.11)
    parser.add_argument("--mean_err",type=float,default=10)
    parser.add_argument("--sigma_err",type=float,default=10)
    parser.add_argument("--nwalkers",type=int,default=12)
    parser.add_argument("--steps",type=int,default=100000)
    parser.add_argument("--sig_strength",type=float,default=10)
    parser.add_argument("--run_bias",type=bool,default=False)
    parser.add_argument("--submit_condor",type=bool,default=False)
    parser.add_argument("--rebin",type=int,default=1)
    parser.add_argument("--ntoys",type=int,default=500)
    parser.add_argument("--tag",default='3j')
    args = parser.parse_args()

    # Load the data
    f = ROOT.TFile(args.input_file)
    h = f.Get(args.hist_name)
    h.Rebin(args.rebin)
    x=[]
    y=[]
    dy=[]
    xc=[]
    yc=[]
    dyc=[]

    for i in range(1,h.GetNbinsX()+1):
        if(h.GetBinContent(i)>-1):
            x.append(h.GetBinCenter(i))
            y.append(h.GetBinContent(i))
            dy.append(h.GetBinErrorUp(i))
            if((h.GetBinCenter(i)<args.mean-2*args.sigma or h.GetBinCenter(i)>args.mean+2*args.sigma)):
                xc.append(h.GetBinCenter(i))
                yc.append(h.GetBinContent(i))
                dyc.append(h.GetBinErrorUp(i))
            
    x = np.array(x)
    y = np.array(y)
    dy = np.array(dy)
    xc = np.array(xc)
    yc = np.array(yc)
    dyc = np.array(dyc)
    dx = x[1]-x[0]

    # Log regularise data
    y_log = np.log(y)
    dy_log = dy/y
    yc_log = np.log(yc)
    dyc_log = dyc/yc

    print("Initial parameters: ",args.variance,args.length_scale," For signal at ",args.mean," with sigma ",args.sigma)
    params = [np.log(args.variance),np.log(args.length_scale)]
    print("Initial parameters (log): ",params)
    result = MinuitMinimize(gp_lml, params, args=(xc[:,None],yc_log[:,None],dyc_log,exponentiated_quadratic))
    # Using a SciPy optimizer
    #from scipy.optimize import minimize
    #result = minimize(gp_lml, params, args=(xc[:,None],(yc_log)[:,None],dyc_log,exponentiated_quadratic)) 

    print(result)

    params = result.x
    parm_result_log = result.x
    cov_result_log = result.hess_inv
    parm_errors_log = np.sqrt(np.diagonal(cov_result_log))
    print("Optimized parameters (linear): ",np.exp(parm_result_log))
    print("Optimized parameters w/ Errors (log): ",parm_result_log[0],'+/-',parm_errors_log[0],',',parm_result_log[1],'+/-',parm_errors_log[1])

    ################ Plotting the posterior #####################
    # Note: The y values are passed in log scale
    # Get the posterior mean and sigma (initially in log)
    x_ = np.linspace(x[0], x[-1], 1000)
    yp, sigma_p = GP_noise(parm_result_log, xc[:,None], yc_log[:,None], x[:,None], exponentiated_quadratic, dyc_log)
    yp = yp.flatten()
    sigma_p = sigma_p.flatten()
    show_fit(x, y_log, dy_log, xc, yc_log, dyc_log, yp, sigma_p, args.mean, args.sigma, tag='fit_test_log_'+args.tag+'_', yscale='linear')
    show_fit(x, y, dy, xc, yc, dyc, np.exp(yp), sigma_p*np.exp(yp), args.mean, args.sigma, tag='fit_test_'+args.tag+'_', yscale='log')











