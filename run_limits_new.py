# import ROOT
# ROOT.EnableImplicitMT()
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
# params={
#     'xtick.labelsize': 'x-small',
#     'ytick.labelsize': 'x-small',
# }
# plt.rcParams['xtick.labelsize'] = 'xx-small'
# plt.rcParams['ytick.labelsize'] = 'xx-small'
cols = ['#4285f4','#ea4335','#fbbc05','#34a853', '#a00498', '#536267']
from prep_limit_new import *
import emcee

def prior(theta,expected_mean,expected_mean_err,expected_sigma,expected_sigma_err,expected_ruc_err):
    amplitude, mean, sigma, ruc = theta
    if amplitude < 0 :
        return 0.0
    if not(expected_mean-1*expected_mean_err<mean<expected_mean+1*expected_mean_err):
        return 0.0
    if not(expected_sigma-1*expected_sigma_err<sigma<expected_sigma+1*expected_sigma_err):
        return 0.0
    if(ruc<=0):
        return 0.0
    prior_mean = np.exp(-0.5*np.power((mean - expected_mean)/expected_mean_err, 2.))/(expected_mean_err * np.sqrt(2*np.pi))
    prior_sigma = np.exp(-0.5*np.power((sigma - expected_sigma)/expected_sigma_err, 2.))/(expected_sigma_err * np.sqrt(2*np.pi))
    prior_ruc = np.exp(-0.5*np.power((np.log(ruc))/expected_ruc_err, 2.))/(ruc*expected_ruc_err * np.sqrt(2*np.pi))
    return prior_ruc*prior_mean*prior_sigma

def log_likelihood(theta,x,y,dy,yp,variance,length_scale,expected_mean,expected_mean_err,expected_sigma,expected_sigma_err,expected_ruc_err):
    amplitude, mean, sigma, ruc = theta
    pri = prior(theta,expected_mean,expected_mean_err,expected_sigma,expected_sigma_err,expected_ruc_err)
    if pri<=0:
        return -np.inf
    else:
        lp = np.log(pri)
    if not np.isfinite(lp):
        return -np.inf
    par_vec = np.array([variance,length_scale,ruc*amplitude,mean,sigma])
    signal = gauss_array(x,mean,sigma,x[1]-x[0])
    mask1 = x<(mean-2*sigma)
    mask2 = x>(mean+2*sigma)
    mask = np.logical_or(mask1,mask2)
    # print(mask)
    xc= x[mask]
    yc= y[mask]
    dyc= dy[mask]
    params = np.array([variance,length_scale])
    yp,_=GP_noise(params,x[:,None],(y-(amplitude*ruc*signal))[:,None],x[:,None],exponentiated_quadratic,dy)
    Mu =  yp.flatten()+(amplitude*ruc*signal)
    # print(xc.shape,yp.shape,Mu.shape)
    mask0=y>0
    y=y[mask0]
    Mu=Mu[mask0]
    Mu[Mu<=0]=1e-3
    ll = -2*np.sum(Mu-y+(y*np.log(y/Mu)))
    # print(ll,np.sum(((y-Mu)/dy)**2))
    # lml = -1.0*gp_sig_lml(par_vec,x, y, dy)
    # lml= lml.flatten()[0]
    # print(lml.shape,ll.shape)
    # print(theta,lp,ll)
    return lp+ll

def likelihood_ratio(theta,x,y,dy,variance,length_scale):
    amplitude, mean, sigma, ruc = theta
    signal = gauss_array(x,mean,sigma,x[1]-x[0])
    mask1 = x<(mean-2*sigma)
    mask2 = x>(mean+2*sigma)
    mask = np.logical_or(mask1,mask2)
    # print(mask)
    xc= x[mask]
    yc= y[mask]
    dyc= dy[mask]
    params = np.array([variance,length_scale])
    yp,_=GP_noise(params,x[:,None],(y-(amplitude*ruc*signal))[:,None],x[:,None],exponentiated_quadratic,dy)
    yp2,_=GP_noise(params,x[:,None],y[:,None],x[:,None],exponentiated_quadratic,dy)
    Mu =  yp.flatten()+(amplitude*ruc*signal)
    yp_ = yp.flatten()+1e-9
    # print(xc.shape,yp.shape,Mu.shape)
    mask0=y>0
    y_=y[mask0]
    Mu=Mu[mask0]
    yp_=yp_[mask0]
    Mu[Mu<=0]=1e-3
    yp_[yp_<=0]=1e-3
    ll = 2*np.sum(Mu-y_+(y_*np.log(y_/Mu)))


    Mu =  yp2.flatten()
    yp_ = yp2.flatten()+1e-9
    # print(xc.shape,yp2.shape,Mu.shape)
    mask0=y>0
    y_=y[mask0]
    Mu=Mu[mask0]
    yp_=yp_[mask0]
    Mu[Mu<=0]=1e-3
    yp_[yp_<=0]=1e-3
    ll2 = 2*np.sum(Mu-y_+(y_*np.log(y_/Mu)))
    # print(ll2-ll)
    return ll2-ll , ll, ll2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--toyn", type=int)
    parser.add_argument("--input_file")
    
    parser.add_argument("--length_scale",type=float,default=100)
    parser.add_argument("--variance",type=float,default=10)
    
    parser.add_argument("--length_scale_err",type=float,default=1)
    parser.add_argument("--variance_err",type=float,default=0.1)
    
    parser.add_argument("--mean",type=float,default=10)
    parser.add_argument("--sigma",type=float,default=10)
    parser.add_argument("--rate_uc",type=float,default=10)
    
    parser.add_argument("--mean_err",type=float,default=10)
    parser.add_argument("--sigma_err",type=float,default=10)
    
    parser.add_argument("--nwalkers",type=int,default=10)
    parser.add_argument("--steps",type=int,default=10000)
    parser.add_argument("--show_result",type=bool,default=False)
    
    parser.add_argument("--sig_strength",type=float,default=10)
    parser.add_argument("--interactive",type=bool,default=False)
    
    args = parser.parse_args() 

    npzfile = np.load(args.input_file)
    x = npzfile['x']
    toys = npzfile['toys']
    yp = npzfile['pred'].flatten()
    print(np.sum(1000.0*gauss_array(x,args.mean,args.sigma,x[1]-x[0]))-1000.0,np.sum(gauss_array(x,args.mean,args.sigma,x[1]-x[0])))

    try:
        y = toys[args.toyn]#+1000.0*gauss_array(x,args.mean,args.sigma,x[1]-x[0])
    except:
        print("Running on OBS")
        y = npzfile['y']

    # print(np.sum(toys[args.toyn])-np.sum(y))

    y=y.astype(float)
    dy= abs((chi2.ppf((1 + 0.681)/2,2*(y.flatten()+1))/2.)-y)
    pos = np.array([args.sig_strength, args.mean, args.sigma, 1])+ np.array([args.sig_strength, args.mean, args.sigma, 1])*1e-4* np.random.randn(args.nwalkers, 4)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(x,y,dy,yp,args.variance,args.length_scale,args.mean,args.mean*args.mean_err,args.sigma,args.sigma*args.sigma_err,args.rate_uc))
    
    if(args.interactive):
        from multiprocessing import Pool
        with Pool(10) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(x,y,dy,yp,args.variance,args.length_scale,args.mean,args.mean*args.mean_err,args.sigma,args.sigma*args.sigma_err,args.rate_uc), pool=pool,moves=emcee.moves.WalkMove())#, moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2),])
            print("Running first burn-in...")
            pos, lp, _ = sampler.run_mcmc(pos, 1000,progress=True)

            pars = pos[np.argmax(lp)]
            # print(pars)
            # print(pars[0]*pars[3]*gauss_array(x,pars[1],pars[2],x[1]-x[0]))
            mask1 = x<(pars[1]-2*pars[2])
            mask2 = x>(pars[1]+2*pars[2])
            mask = np.logical_or(mask1,mask2)
            xc= x#[mask]
            yc= y#[mask]
            dyc= dy#[mask]
            parm=np.array([args.variance,args.length_scale])
            if(args.show_result): show_fit(parm, x, y, dy, xc, yc, dyc , pars[1], pars[2], None,pars[0]*pars[3],'toy_br_sig_',yp)

            print("Running second burn-in...")
            print('Max lp',np.max(lp),prior(pos[np.argmax(lp)],args.mean, args.mean*args.mean_err, args.sigma, args.sigma*args.sigma_err, args.rate_uc))
            print('Max lp @',pos[np.argmax(lp)])
            pos = pos[np.argmax(lp)] + 1e-2 * np.random.randn(nwalkers, ndim)
            sampler.reset()
            pos, lp, _ = sampler.run_mcmc(pos, 1000,progress=True)
            pos = pos[np.argmax(lp)] + 1e-2 * np.random.randn(nwalkers, ndim)
            print('Max lp',np.max(lp),np.max(lp),prior(pos[np.argmax(lp)],args.mean, args.mean*args.mean_err, args.sigma, args.sigma*args.sigma_err, args.rate_uc))
            print('Max lp @',pos[np.argmax(lp)])
            sampler.reset()

            print("Running production...")
            pos, lp, _ = sampler.run_mcmc(pos, args.steps,progress=True)
            print('Max lp',np.max(lp),prior(pos[np.argmax(lp)],args.mean, args.mean*args.mean_err, args.sigma, args.sigma*args.sigma_err, args.rate_uc))
            print('Max lp @',pos[np.argmax(lp)])
            tau = sampler.get_autocorr_time()
            print("Auto Correlation time",tau)
            if(args.show_result): show_fit(parm, x, y, dy, xc, yc, dyc , pars[1], pars[2], None,pars[0]*pars[3],'toy_br_sig2_',yp)

            pars = pos[np.argmax(lp)]
            lhr,llsb,llb=likelihood_ratio(pars,x,y,dy,args.variance,args.length_scale)
            mask1 = x<(pars[1]-2*pars[2])
            mask2 = x>(pars[1]+2*pars[2])
            mask = np.logical_or(mask1,mask2)
            xc= x[mask]
            yc= y[mask]
            dyc= dy[mask]
            parm=np.array([args.variance,args.length_scale])
            if(args.show_result): show_fit(parm, x, y, dy, xc, yc, dyc , pars[1], pars[2], None,pars[0]*pars[3],'toy_pf_sig_',yp)

    else:
        print("Running first burn-in...")
        pos, lp, _ = sampler.run_mcmc(pos, 1000,progress=True)
        print("Running second burn-in...")
        pos = pos[np.argmax(lp)] + 1e-2 * np.random.randn(nwalkers, ndim)
        sampler.reset()
        print('Max lp',np.max(lp),prior(pos[np.argmax(lp)],args.mean, args.mean*args.mean_err, args.sigma, args.sigma*args.sigma_err, args.rate_uc))
        print('Max lp @',pos[np.argmax(lp)])
        pos, lp, _ = sampler.run_mcmc(pos, 2000,progress=True)
        pos = pos[np.argmax(lp)] + 1e-2 * np.random.randn(nwalkers, ndim)
        print('Max lp',np.max(lp),np.max(lp),prior(pos[np.argmax(lp)],args.mean, args.mean*args.mean_err, args.sigma, args.sigma*args.sigma_err, args.rate_uc))
        print('Max lp @',pos[np.argmax(lp)])
        sampler.reset()

        print("Running production...")
        pos, lp, _ = sampler.run_mcmc(pos, args.steps,progress=True)
        print('Max lp',np.max(lp),prior(pos[np.argmax(lp)],args.mean, args.mean*args.mean_err, args.sigma, args.sigma*args.sigma_err, args.rate_uc))
        print('Max lp @',pos[np.argmax(lp)])
        tau = sampler.get_autocorr_time()
        print("Auto Correlation time",tau)
        
        pars = pos[np.argmax(lp)]
        lhr,llsb,llb=likelihood_ratio(pars,x,y,dy,args.variance,args.length_scale)




    samples = sampler.get_chain(discard=int(args.steps/4), flat=False)
    print(samples.shape)
    
    cl_lims=[]
    for i in range(nwalkers):
        cl_lims.append(np.quantile(samples[:,i,0],0.95))
        print("cl with walker {}:".format(i),np.quantile(samples[:,i,0],0.95))
    
    cl_lims=np.array(cl_lims)
    
    flat_samples = sampler.get_chain(discard=int(args.steps/4), flat=True)
    print(flat_samples.shape)
    
    amp_95 = np.mean(cl_lims)
    amp_95_err = np.std(cl_lims)
    
    print('amplitude error @ 95: ', amp_95_err)
    print('amplitude @ 95: ', amp_95)
    print('likelihood_ratio @ MAP: ',lhr)
    print(np.quantile(flat_samples[:,0],0.95))
    
    if(args.interactive):
        import corner
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["A", "\u03BC", "\u03C3", "ruc"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.savefig('figs/'+args.input_file.split(".npz")[0]+'_walkers_'+str(args.toyn)+'.png')
        # plt.show()
        
#         flat_samples = sampler.get_chain(discard=int(args.steps/5), flat=True)

#         fig = corner.corner(
#             flat_samples, labels=labels, truths=[1000, 500, 40, 1]
#         );
        
        amp_vals = flat_samples[:,0]
        fit_mean, fit_sigma =norm.fit(amp_vals)
        print("##### Values from AMP fit: mean {} and sigma {}".format(fit_mean, fit_sigma))
        print(flat_samples[:,:3].shape)
        
    
        fig = corner.corner( flat_samples[:,:3], labels=labels[:3],label_kwargs={"fontsize": 20})#, truths=[amp_95, args.mean, args.sigma]);
        print("hi")
        corner.overplot_lines(fig, [None, np.mean(flat_samples[:,1]), np.mean(flat_samples[:,2])], color="#4285F4")
        print("hello")
        corner.overplot_lines(fig, [amp_95, None, None], color="#FBBC05")
        print("hello there")
        values = np.array([[amp_95, np.mean(flat_samples[:,1]), np.mean(flat_samples[:,2])]])
        corner.overplot_points(fig, values,marker="s", color="#34A853")
        fig.savefig('figs/'+args.input_file.split(".npz")[0]+'_'+str(args.toyn)+'.pdf')
        
#         fig = corner.corner()
#             flat_samples, labels=labels, truths=[0, 500, 40, 1]
#         )
#         plt.savefig(args.input_file.split(".npz")[0]+'_'+str(args.toyn)+'.png')
    print(np.quantile(flat_samples[:,0],0.5))
    print(np.mean(flat_samples[:,0]))
    print(np.std(flat_samples[:,0]))
    print(np.quantile(flat_samples[:,0],0.95))

    
    for i in range(3):
        for j in range(i+1,3):
            print("r({},{})".format(labels[i],labels[j]),np.corrcoef(flat_samples[:,i],flat_samples[:,j]))


