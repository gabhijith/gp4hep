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

# jax.config.update("jax_enable_x64", True)

# @jax.jit
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
    """
    K11 = kernel_func(X1, X1,np.exp(params[0]),np.exp(params[1])) + np.diag(noise**2)
    K12 = kernel_func(X1, X2,np.exp(params[0]),np.exp(params[1]))
    solved = scipy.linalg.solve(K11, K12, assume_a='pos').T
    μ2 = solved @ y1
    K22 = kernel_func(X2, X2,np.exp(params[0]),np.exp(params[1]))
    K2 = K22 - (solved @ K12)
    return μ2, np.sqrt(np.diag(K2))

# @jax.jit
def gauss_array(x, mu=100, sig=10, bw=1):
#     print(x.shape,mu,sig,bw)
    y = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    den = sig * ((2*np.pi)**0.5)
    y=y*bw/den
    # assert np.around(np.sum(y)*(x[1]-x[0]))==bw, (np.sum(y), "is not 1", mu, sig, bw)
    return y

# @jax.jit
def gp_lml(params,X, y, noise, kernel_func):
    K11 = kernel_func(X, X,np.exp(params[0]),np.exp(params[1])) + np.diag(noise**2)
    return 0.5*(y.T @ np.linalg.inv(K11) @ y) + 0.5*np.log(np.linalg.det(K11)) + 0.5*np.float64(y.shape[0])*np.log(2*np.pi)

# @jax.jit
def gp_sig_lml(params,x, y, noise):
    # print(params)
    K11 = exponentiated_quadratic(x[:,None], x[:,None],np.exp(params[0]),np.exp(params[1])) + np.diag(noise**2)
    sig = params[2]*gauss_array(x,params[3],params[4],x[1]-x[0])
    Y = (y - sig)[:,None]
    lml=0.5*(Y.T @ np.linalg.inv(K11) @ Y) + 0.5*np.log(np.linalg.det(K11)) + 0.5*np.float64(Y.shape[0])*np.log(2*np.pi)
    return lml.flatten()[0]

# @jax.jit
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

def show_fit(params, x, y, dy, xc, yc, dyc , mean, sigma, toys=None,amp=None,tag='fit',yp_=None):
    x_ = np.linspace(x[0], x[-1], 1000)
    y_, sigma_ = GP_noise(params, xc[:,None], yc[:,None], x_[:,None], exponentiated_quadratic,dyc)
    y_ = y_.flatten()
    yp,sigma_p=GP_noise(params,xc[:,None],yc[:,None],x[:,None],exponentiated_quadratic,dyc)
    yp = yp.flatten()
    fig = plt.figure(figsize=(10, 13))
    spec = fig.add_gridspec(ncols=1, nrows=3,height_ratios=[0.8,0.2,0.3])
    main = fig.add_subplot(spec[0,0])
    if toys is not None:
        main.plot(x, toys[:10].T,'.',color='red',alpha=0.3)
    main.errorbar(x, y, dy, fmt='ok', label='Data')
    main.plot(x_, y_, label='Prediction',color=cols[1])
    main.fill_between(x_, y_-sigma_, y_+sigma_, label='68% CI',color=cols[1],alpha=0.2)
    main.axvline(x=mean-2*sigma,color='lightgrey',linestyle='--')
    main.axvline(x=mean+2*sigma,color='lightgrey',linestyle='--')
    main.plot([], [], ' ', label='{}/{}'.format(np.sum(np.square((y-yp)/dy)),y.shape[0]))
    if amp is not None:
        main.plot([], [], ' ', label='{}/{}'.format(np.sum(np.square((y-amp*gauss_array(x,mean,sigma,x[1]-x[0])-yp)/dy)),y.shape[0]))
    main.legend()
    if amp is not None:
        main.plot(x_, amp*gauss_array(x_,mean,sigma,x[1]-x[0]),label='Signal',color=cols[3])
    if yp_ is not None:
        main.plot(x, yp_,'--',color=cols[4],alpha=0.5)
    main.set_ylim(0.5,1000)
    main.set_yscale('log')
    pull = fig.add_subplot(spec[1,0])
    pull.bar(x, (y-yp)/dy, width=x[1]-x[0], label='Data Pull',color=cols[1])
    if amp is not None:
        pull.bar(x, (y-amp*gauss_array(x,mean,sigma,x[1]-x[0])-yp)/dy, width=x[1]-x[0], label='Signal Pull',color=cols[3])
    pull.axvline(x=mean-2*sigma,color='lightgrey',linestyle='--')
    pull.axvline(x=mean+2*sigma,color='lightgrey',linestyle='--')
    pull.set_ylim(-5,5)
    pull.legend()
    diff = fig.add_subplot(spec[2,0])
    diff.errorbar(x, y-yp, dy, fmt='ok', label='Data')
    if yp_ is not None:
        diff.errorbar(x, y-yp_, dy, fmt='.', label='Data',color='lightgrey',alpha=0.5)
    if amp is not None:
        diff.errorbar(x, y-yp-amp*gauss_array(x,mean,sigma,x[1]-x[0]), dy, fmt='.', label='Data',color='lightgrey',alpha=0.5)
    diff.fill_between(x_, -sigma_, sigma_, label='68% CI',color=cols[1],alpha=0.1)
    diff.axvline(x=mean-2*sigma,color='lightgrey',linestyle='--')
    diff.axvline(x=mean+2*sigma,color='lightgrey',linestyle='--')
    diff.axhline(y=0,color='lightgrey',linestyle='-')
    diff.set_ylim(np.min(y-yp-1.1*dy),np.max(y-yp+1.1*dy))
    if amp is not None:
        # diff.plot(x_, amp*gauss_array(x_,mean,sigma,x[1]-x[0]),label='Signal',color=cols[3])
        diff.plot(x,amp*gauss_array(x,mean,sigma,x[1]-x[0]),drawstyle='steps-mid',color=cols[2],label='TTbar MC')
        min_ = np.min(y-yp-1.1*dy)
        max_ = max(np.max(y-yp+1.1*dy),np.max(amp*gauss_array(x_,mean,sigma,x[1]-x[0])))
        diff.set_ylim(min_,max_)
    plt.savefig(tag+str(int(mean))+'.pdf')

def log_likelihood(params,x,y,dy):
    signal = params[2]*gauss_array(x,params[3],params[4],x[1]-x[0])
    mask1 = x<(params[3]-2*params[4])
    mask2 = x>(params[3]+2*params[4])
    mask = np.logical_or(mask1,mask2)
    # print(mask)
    xc= x[mask]
    yc= y[mask]
    dyc= dy[mask]
    yp,_=GP_noise(params[:2],xc[:,None],yc[:,None],x[:,None],exponentiated_quadratic,dyc)
    Mu =  yp.flatten()+signal+1e-9
    yp_ = yp.flatten()+1e-9
    # print(xc.shape,yp.shape,Mu.shape)
    mask0=y>0
    y=y[mask0]
    Mu=Mu[mask0]
    yp_=yp_[mask0]
    Mu[Mu<=0]=1e-3
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
    xc = x#np.array(xc)
    yc = y#np.array(yc)
    dyc = dy#np.array(dyc)

    print(x[1]-x[0])

    print("Initial parameters: ",args.variance,args.length_scale," For signal at ",args.mean," with sigma ",args.sigma)
    params = np.log(np.array([args.variance,args.length_scale]))
    result = MinuitMinimize(gp_lml, params, args=(xc[:,None],(yc)[:,None],dyc,exponentiated_quadratic))
    print(result)
    print("Optimized parameters: ",np.exp(result.x))
    parm_errors=np.sqrt(np.diagonal(result.hess_inv))
    print("Optimized parameters w/ Errors: ",params[0],'+/-',parm_errors[0],',',params[1],'+/-',parm_errors[1])

    params = np.log(np.array([args.variance,args.length_scale]))#result.x
    yp,sigma_p=GP_noise(params,xc[:,None],yc[:,None],x[:,None],exponentiated_quadratic,dyc)
    show_fit(params, x, y, dy, xc, yc, dyc , args.mean, args.sigma, None,None,'fit_'+args.tag+'_')

    ys = sample_gp(params,xc[:,None],yc[:,None],x[:,None],exponentiated_quadratic,dyc,args.ntoys)
    ys[ys<0]=0
    ys_toys = make_toy(ys,h,int(np.sum(yp)))
    show_fit(params, x, y, dy, xc, yc, dyc , args.mean, args.sigma, ys_toys,None,'fit_toy_'+args.tag+'_')
    ys_toys_ws = np.copy(ys_toys)
    
    for i,toy in enumerate(ys_toys_ws): 
        ys_toys_ws[i] = toy+np.around(args.sig_strength*gauss_array(x,args.mean,args.sigma,x[1]-x[0]))
    np.savez('toys_'+str(int(args.mean))+'_'+args.tag+'_3j.npz',toys = ys_toys, x = x, toys_ws = ys_toys_ws, y = y, pred = yp)

    # cost = partial(gp_sig_lml,x=x,y=y,noise=dy)
    cost = partial(log_likelihood,x=x,y=y,dy=dy)
    m = Minuit(cost, (params[0], params[1],args.sig_strength,args.mean,args.sigma), name=("Variance", "length_scale", "sig_strength", "mean", "sigma"))
    m.errordef = 0.5
    # m.fixed[0] = True
    # m.fixed[1] = True
    m.fixed[3] = True
    m.fixed[4] = True
    m.limits[0] = (params[0]-2*parm_errors[0],params[0]+2*parm_errors[0])
    m.limits[1] = (params[1]-2*parm_errors[1],params[1]+2*parm_errors[1])
    m.limits[2] = (0,None)
    m.limits[3] = (args.mean-1*(args.mean_err*args.mean),args.mean+1*(args.mean_err*args.mean))
    m.limits[4] = (args.sigma-1*(args.sigma_err*args.sigma),args.sigma+1*(args.sigma_err*args.sigma)) 
    # m.limits[4] = (5,20)
    m.migrad()
    m.minos()
    m.migrad()
    fig = plt.figure(figsize=(10, 10))
    a, fa = m.profile("sig_strength")
    plt.plot(a, fa)
    plt.savefig('sig_strength.pdf')
    print(m)

    show_fit(params, x, y, dy, xc, yc, dyc , m.values['mean'], m.values['sigma'], None,m.values['sig_strength'],'fit_sig_'+args.tag+'_')
    ss_1sigma = m.values['sig_strength']+2*m.errors['sig_strength']
    ss = m.values['sig_strength']

    print('To run interactively do:')
    print("python run_limits_new.py --toyn={} --input_file={} --length_scale={} --variance={} --length_scale_err={} --variance_err={} --mean={} --sigma={} --rate_uc={} --mean_err={} --sigma_err={} --nwalkers={} --steps={} --sig_strength={} --show_result=True".format(str(1),'toys_'+str(int(args.mean))+'_'+args.tag+'_3j.npz',str(params[1]),str(params[0]),str(0),str(0),str(args.mean),str(args.sigma),str(args.rate_uc),str(args.mean_err),str(args.sigma_err),str(args.nwalkers),str(args.steps),str(m.values["sig_strength"]),int(args.mean)))

    # submit = input("Submit to condor Y/N: ")

    print("submit to condor",args.submit_condor)
    if(args.submit_condor):
        print("Submitting to Condor")
        os.system('rm run_limits_'+str(int(args.mean))+'.jcl')
        os.system('cp run_limits.jcl run_limits_'+str(int(args.mean))+'.jcl')

        os.system('sed -i -e "s/**input_file/{}/g" run_limits_{}.jcl'.format('toys_'+str(int(args.mean))+'_'+args.tag+'_3j.npz',int(args.mean)))
        os.system('sed -i -e "s/**length_scale_err/{}/g" run_limits_{}.jcl'.format(0,int(args.mean)))
        os.system('sed -i -e "s/**variance_err/{}/g" run_limits_{}.jcl'.format(0,int(args.mean)))
        os.system('sed -i -e "s/**length_scale/{}/g" run_limits_{}.jcl'.format(params[1],int(args.mean)))
        os.system('sed -i -e "s/**variance/{}/g" run_limits_{}.jcl'.format(params[0],int(args.mean)))
        os.system('sed -i -e "s/**mean_err/{}/g" run_limits_{}.jcl'.format(str(args.mean_err),int(args.mean)))
        os.system('sed -i -e "s/**sigma_err/{}/g" run_limits_{}.jcl'.format(str(args.sigma_err),int(args.mean)))
        os.system('sed -i -e "s/**mean/{}/g" run_limits_{}.jcl'.format(str(int(args.mean)),int(args.mean)))
        os.system('sed -i -e "s/**sigma/{}/g" run_limits_{}.jcl'.format(str(args.sigma),int(args.mean)))
        os.system('sed -i -e "s/**rate_uc/{}/g" run_limits_{}.jcl'.format(str(args.rate_uc),int(args.mean)))
        os.system('sed -i -e "s/**nwalkers/{}/g" run_limits_{}.jcl'.format(str(args.nwalkers),int(args.mean)))
        os.system('sed -i -e "s/**steps/{}/g" run_limits_{}.jcl'.format(str(args.steps),int(args.mean)))
    #     os.system('sed -i -e "s/**steps/{}/g" run_limits_{}.jcl'.format(str(args.steps),int(args.mean)))
        os.system('sed -i -e "s/**sig_strength/{}/g" run_limits_{}.jcl'.format(m.values["sig_strength"],int(args.mean)))
        os.system('sed -i -e "s/0000/{}/g" run_limits_{}.jcl'.format(str(args.ntoys+1),int(args.mean)))
        os.system('sed -i -e "s/**tag/{}/g" run_limits_{}.jcl'.format(args.tag,int(args.mean)))
        
        os.system('cat run_limits_'+str(int(args.mean))+'.jcl')
        
        
        os.system('condor_submit run_limits_'+str(int(args.mean))+'.jcl')
        print()
        print("submitted limits and reverted the file back")
        print()
        os.system('cat run_limits.jcl')


    if(args.run_bias):
        print("Running Bias")
        for lm in [1.0]:
            if lm>1.0: mult=lm-1
            else: mult = lm
            if ss_1sigma<args.sig_strength: ss_1sigma=args.sig_strength
            print("Running for lm = ",lm)
            sig_strength=np.ones(args.ntoys)*m.values['sig_strength']
            sig_error=np.zeros(args.ntoys)
            for i in range(args.ntoys):
                if(i%100==0):print("@",i)
                Y=ys_toys[i] + np.around(mult*ss_1sigma*gauss_array(x,m.values['mean'],m.values['sigma'],x[1]-x[0]))
                DY=abs((chi2.ppf((1 + 0.681)/2,2*(Y.flatten()+1))/2.)-Y)
                cost = partial(log_likelihood,x=x,y=Y,dy=DY)
                m = Minuit(cost, (params[0], params[1],ss,args.mean,args.sigma), name=("Variance", "length_scale", "sig_strength", "mean", "sigma"))
                m.errordef = 0.5
                m.fixed[0] = True
                m.fixed[1] = True
                m.fixed[3] = True
                m.fixed[4] = True
                m.limits[2] = (0,None)
                m.migrad()
                sig_strength[i]=m.values['sig_strength']
                mask1 = x<(args.mean-2*args.sigma)
                mask2 = x>(args.mean+2*args.sigma)
                mask = np.logical_or(mask1,mask2)
                # print(mask)
                xc= x[mask]
                Yc= Y[mask]
                DYc= DY[mask]
                # show_fit(params, x, Y, DY, xc, Yc, DYc , m.values['mean'], m.values['sigma'], None,m.values['sig_strength'],'bias_figs/bias3_'+str(int(args.mean))+'_'+str(int(lm))+'_'+str(i)+'3j.pdf')
            bias_array = sig_strength - np.sum(np.around(mult*ss_1sigma*gauss_array(x,m.values['mean'],m.values['sigma'],x[1]-x[0])))
            bias_array = bias_array/np.std(sig_strength)
            fig = plt.figure(figsize=(8, 6))
            # bins = np.linspace(-5,5,20)
            mean,std=norm.fit(bias_array)
            print("Mean: ",mean," Std: ",std)
            print("Mean: ",np.mean(bias_array)," Std: ",np.std(bias_array))
            print()
            # plt.title("Bias @ "+str(int(lm))+" sigma, "+str(int(args.mean))+" GeV")
            plt.hist(bias_array,bins=20, density=True)
            xmin, xmax = plt.xlim()
            xx = np.linspace(xmin, xmax, 100)
            yy = norm.pdf(xx, mean, std)
            plt.plot(xx, yy,label=r"$\mu$: {0:.2f}," "\n" r"$\sigma$ : {1:.2f}".format(mean,std,int(args.mean)))
            plt.xlim(-4.5,4.5)
            plt.xlabel(r'$Z_{W}$')
            plt.ylabel('Fraction')
            plt.legend()
            plt.savefig('bias3_'+str(int(args.mean))+'_'+str(int(lm))+'_3j.pdf', bbox_inches="tight")

            fig = plt.figure(figsize=(8, 6))
            bias_array = sig_strength - np.sum(np.around(mult*ss_1sigma*gauss_array(x,m.values['mean'],m.values['sigma'],x[1]-x[0])))
            mean,std=norm.fit(bias_array)
            print("Mean: ",mean," Std: ",std)
            print("Mean: ",np.mean(bias_array)," Std: ",np.std(bias_array))
            print()
            plt.hist(bias_array,bins=20, density=True)
            xmin, xmax = plt.xlim()
            xx = np.linspace(xmin, xmax, 100)
            yy = norm.pdf(xx, mean, std)
            plt.plot(xx, yy,label=r"$\mu$: {0:.2f}," "\n" r"$\sigma$ : {1:.2f}".format(mean,std,int(args.mean)))
            plt.xlim(-4.5*std,4.5*std)
            plt.xlabel(r'$\Delta A$')
            plt.ylabel('Fraction')
            plt.legend()
            plt.savefig('ss3_'+str(int(args.mean))+'_'+str(int(lm))+'_3j.pdf', bbox_inches="tight")






