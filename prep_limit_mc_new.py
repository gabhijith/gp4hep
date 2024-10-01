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
plt.style.use(hep.style.ROOT)
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
    return y

# @jax.jit
def gp_lml(params,X, y, noise, kernel_func):
    K11 = kernel_func(X, X,np.exp(params[0]),np.exp(params[1])) + np.diag(noise**2)
    return 0.5*(y.T @ np.linalg.inv(K11) @ y) + 0.5*np.log(np.linalg.det(K11)) + 0.5*np.float64(y.shape[0])*np.log(2*np.pi)

# @jax.jit
def gp_sig_lml(params,x, y, noise,ym):
    # print(params)
    K11 = exponentiated_quadratic(x[:,None], x[:,None],np.exp(params[0]),np.exp(params[1])) + np.diag(noise**2)
    sig = params[2]*gauss_array(x,params[3],params[4],x[1]-x[0])
    Y = (y - sig-ym)[:,None]
    return 0.5*(Y.T @ np.linalg.inv(K11) @ Y) + 0.5*np.log(np.linalg.det(K11)) + 0.5*np.float64(Y.shape[0])*np.log(2*np.pi)

def log_likelihood(params,x,y,dy,ym):
    signal = params[2]*gauss_array(x,params[3],params[4],x[1]-x[0])
    mask1 = x<(params[3]-2*params[4])
    mask2 = x>(params[3]+2*params[4])
    mask = np.logical_or(mask1,mask2)
    # print(mask)
    xc= x[mask]
    yc= y[mask]
    dyc= dy[mask]
    ycm = ym[mask]
    yp,_=GP_noise(params,xc[:,None],(yc-ycm)[:,None],x[:,None],exponentiated_quadratic,dyc)
    Mu =  yp.flatten()+signal+ym.flatten()+1e-9
    # print(xc.shape,yp.shape,Mu.shape)
    mask0=y>0
    y=y[mask0]
    Mu=Mu[mask0]
    ll = 2*np.sum(Mu-y+(y*np.log(y/Mu)))
    return ll

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

def show_fit(params, x, y, dy, xc, yc, dyc ,ym,ycm, mean, sigma, toys=None,amp=None,tag='fit',yp_=None):
    x_ = np.linspace(x[0], x[-1], 1000)
    y_, sigma_ = GP_noise(params, xc[:,None], (yc-ycm)[:,None], x_[:,None], exponentiated_quadratic,dyc)
    y_ = y_.flatten()
    yp,sigma_p=GP_noise(params,xc[:,None],(yc-ycm)[:,None],x[:,None],exponentiated_quadratic,dyc)
    yp = yp.flatten()
    fig = plt.figure(figsize=(10, 13))
    spec = fig.add_gridspec(ncols=1, nrows=3,height_ratios=[0.8,0.2,0.3])
    main = fig.add_subplot(spec[0,0])
    if toys is not None:
        main.plot(x, toys[:10].T,'.',color='red',alpha=0.3)
    main.errorbar(x, y, dy, fmt='ok', label='Data')
    main.plot(x_, y_, label='Prediction',color=cols[1])
    plt.plot(x,yp+ym,drawstyle='steps-mid',color=cols[2],label='Prediction w/ MC')
    main.fill_between(x_, y_-sigma_, y_+sigma_, label='68% CI',color=cols[1],alpha=0.2)
    main.axvline(x=mean-2*sigma,color='lightgrey',linestyle='--')
    main.axvline(x=mean+2*sigma,color='lightgrey',linestyle='--')
    main.legend()
    if amp is not None:
        main.plot(x_, amp*gauss_array(x_,mean,sigma,x[1]-x[0]),label='Signal',color=cols[3])
    if yp_ is not None:
        main.plot(x, yp_,'--',color=cols[4],alpha=0.5)
    pull = fig.add_subplot(spec[1,0])
    pull.bar(x, (y-yp-ym)/dy, width=x[1]-x[0], label='Data Pull',color=cols[1])
    pull.axvline(x=mean-2*sigma,color='lightgrey',linestyle='--')
    pull.axvline(x=mean+2*sigma,color='lightgrey',linestyle='--')
    pull.set_ylim(-5,5)
    pull.legend()
    diff = fig.add_subplot(spec[2,0])
    diff.errorbar(x, y-yp-ym, dy, fmt='ok', label='Data')
    if yp_ is not None:
        diff.errorbar(x, y-yp_-ym, dy, fmt='.', label='Data',color='lightgrey',alpha=0.5)
    diff.fill_between(x_, -sigma_, sigma_, label='68% CI',color=cols[1],alpha=0.1)
    diff.axvline(x=mean-2*sigma,color='lightgrey',linestyle='--')
    diff.axvline(x=mean+2*sigma,color='lightgrey',linestyle='--')
    diff.axhline(y=0,color='lightgrey',linestyle='-')
    diff.set_ylim(np.min(y-yp-ym-1.1*dy),np.max(y-yp-ym+1.1*dy))
    if amp is not None:
        diff.plot(x_, amp*gauss_array(x_,mean,sigma,x[1]-x[0]),label='Signal',color=cols[3])
        min_ = np.min(y-yp-ym-1.1*dy)
        max_ = max(np.max(y-yp-ym+1.1*dy),np.max(amp*gauss_array(x_,mean,sigma,x[1]-x[0])))
        diff.set_ylim(min_,max_)
    plt.savefig(tag+str(int(mean))+'.pdf')


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
    parser.add_argument("--mc_uc",type=float,default=0.1)
    parser.add_argument("--tag",default='3j')
    args = parser.parse_args()

    # Load the data
    file = ROOT.TFile(args.input_file)
    h = file.Get(args.hist_name)
    hm = file.Get(args.hist_name+'_mc')


    x=[]
    y=[]
    dy=[]

    xc=[]
    yc=[]
    dyc=[]

    xm=[]
    ym=[]
    dym=[]

    xcm=[]
    ycm=[]
    dycm=[]

    h.SetBinErrorOption(ROOT.TH1.kPoisson)
    hm.SetBinErrorOption(ROOT.TH1.kPoisson)
    print(h.GetNbinsX())    
    for i in range(1,h.GetNbinsX()+1):
            
            x.append(h.GetBinCenter(i))
            y.append(h.GetBinContent(i))
            dy.append(h.GetBinErrorUp(i))
            
            xm.append(hm.GetBinCenter(i))
            ym.append(hm.GetBinContent(i))
            dym.append(hm.GetBinErrorUp(i))
            
            if((h.GetBinCenter(i)<args.mean-2*args.sigma or h.GetBinCenter(i)>args.mean+2*args.sigma)):
                xc.append(h.GetBinCenter(i))
                yc.append(h.GetBinContent(i))
                dyc.append(h.GetBinErrorUp(i))
                
                xcm.append(hm.GetBinCenter(i))
                ycm.append(hm.GetBinContent(i))
                dycm.append(hm.GetBinErrorUp(i))
            
    x = np.array(x)
    y = np.array(y)
    dy = np.array(dy)
    xc = np.array(xc)
    yc = np.array(yc)
    dyc = np.array(dyc)
    xm=np.array(xm)
    ym=np.array(ym)
    dym=np.array(dym)
    xcm=np.array(xcm)
    ycm=np.array(ycm)
    dycm=np.array(dycm)

    print(x.shape,y.shape,dy.shape,xc.shape,yc.shape,dyc.shape,xm.shape,ym.shape,dym.shape,xcm.shape,ycm.shape,dycm.shape)
    print("Initial parameters: ",args.variance,args.length_scale," For signal at ",args.mean," with sigma ",args.sigma)
    params = np.log(np.array([args.variance,args.length_scale]))
    result = MinuitMinimize(gp_lml, params, args=(xc[:,None],(yc-ycm)[:,None],dyc,exponentiated_quadratic))
    print(result)
    print("Optimized parameters: ",np.exp(result.x))

    params = result.x
    yp,sigma_p=GP_noise(params,xc[:,None],(yc-ycm)[:,None],x[:,None],exponentiated_quadratic,dyc)
    show_fit(params, x, y, dy, xc, yc, dyc,ym,ycm , args.mean, args.sigma, None,None,'fit_')

    ys = sample_gp(params,xc[:,None],(yc-ycm)[:,None],x[:,None],exponentiated_quadratic,dyc,500)
    ys[ys<0]=0
    ys_toys = make_toy(ys,h,int(np.sum(yp)))
    ys_toys_ws = np.copy(ys_toys)
    
    for i,toy in enumerate(ys_toys_ws): 
        ys_toys_ws[i] = toy+np.around(args.sig_strength*gauss_array(x,args.mean,args.sigma,x[1]-x[0]))+np.around(ym)
        ys_toys[i]=ys_toys[i]+np.around(ym)
    show_fit(params, x, y, dy, xc, yc, dyc,ym,ycm , args.mean, args.sigma, ys_toys,None,'fit_toy_')
    np.savez('toys_'+str(int(args.mean))+'_'+args.tag+'_3j.npz',toys = ys_toys, x = x, toys_ws = ys_toys_ws, y = y, pred = yp, ym=ym)

    cost = partial(log_likelihood,x=x,y=y,dy=dy,ym=ym)
    m = Minuit(cost, (params[0], params[1],args.sig_strength,args.mean,args.sigma), name=("Variance", "length_scale", "sig_strength", "mean", "sigma"))
    m.errordef = 0.5
    m.fixed[0] = True
    m.fixed[1] = True
    m.fixed[3] = True
    m.fixed[4] = True
    m.limits[2] = (0,None)
    m.limits[3] = (args.mean-2*(args.mean_err*args.mean),args.mean+2*(args.mean_err*args.mean))
    m.limits[4] = (args.sigma-2*(args.sigma_err*args.sigma),args.sigma+2*(args.sigma_err*args.sigma))
    m.migrad()
    fig = plt.figure(figsize=(10, 10))
    a, fa = m.profile("sig_strength")
    plt.plot(a, fa)
    plt.savefig('sig_strength.pdf')
    print(m)

    show_fit(params, x, y, dy, xc, yc, dyc,ym,ycm , m.values['mean'], m.values['sigma'], None,m.values['sig_strength'],'fit_sig_')
    ss_1sigma = m.values['sig_strength']+2*m.errors['sig_strength']
    ss = m.values['sig_strength']
    ss_sigma = m.errors['sig_strength']

    print('To run interactively do:')
    print("python run_limits_mc_new.py --toyn={} --input_file={} --length_scale={} --variance={} --length_scale_err={} --variance_err={} --mean={} --sigma={} --rate_uc={} --mean_err={} --sigma_err={} --nwalkers={} --steps={} --sig_strength={} --show_result=True".format(str(1),'toys_'+str(int(args.mean))+'_3j.npz',str(params[1]),str(params[0]),str(0),str(0),str(args.mean),str(args.sigma),str(args.rate_uc),str(args.mean_err),str(args.sigma_err),str(args.nwalkers),str(args.steps),str(m.values["sig_strength"])))

    # submit = input("Submit to condor Y/N: ")

    print("submit to condor",args.submit_condor)
    if(args.submit_condor):
        print("Submitting to Condor")
        os.system('rm run_limits_mc_'+str(int(args.mean))+'.jcl')
        os.system('cp run_limits_mc.jcl run_limits_mc_'+str(int(args.mean))+'.jcl')

        os.system('sed -i -e "s/**input_file/{0}/g" run_limits_mc_{1}.jcl'.format('toys_'+str(int(args.mean))+'_'+args.tag+'_3j.npz',int(args.mean)))
        os.system('sed -i -e "s/**length_scale_err/{0}/g" run_limits_mc_{1}.jcl'.format(0,int(args.mean)))
        os.system('sed -i -e "s/**variance_err/{0}/g" run_limits_mc_{1}.jcl'.format(0,int(args.mean)))
        os.system('sed -i -e "s/**length_scale/{0}/g" run_limits_mc_{1}.jcl'.format(params[1],int(args.mean)))
        os.system('sed -i -e "s/**variance/{0}/g" run_limits_mc_{1}.jcl'.format(params[0],int(args.mean)))
        os.system('sed -i -e "s/**mean_err/{0}/g" run_limits_mc_{1}.jcl'.format(str(args.mean_err),int(args.mean)))
        os.system('sed -i -e "s/**sigma_err/{0}/g" run_limits_mc_{1}.jcl'.format(str(args.sigma_err),int(args.mean)))
        os.system('sed -i -e "s/**mean/{0}/g" run_limits_mc_{1}.jcl'.format(str(int(args.mean)),int(args.mean)))
        os.system('sed -i -e "s/**sigma/{0}/g" run_limits_mc_{1}.jcl'.format(str(args.sigma),int(args.mean)))
        os.system('sed -i -e "s/**rate_uc/{0}/g" run_limits_mc_{1}.jcl'.format(str(args.rate_uc),int(args.mean)))
        os.system('sed -i -e "s/**mc_uc/{0}/g" run_limits_mc.jcl'.format(str(args.mc_uc),int(args.mean)))
        os.system('sed -i -e "s/**nwalkers/{0}/g" run_limits_mc_{1}.jcl'.format(str(args.nwalkers),int(args.mean)))
        os.system('sed -i -e "s/**steps/{0}/g" run_limits_mc_{1}.jcl'.format(str(args.steps),int(args.mean)))
    #     os.system('sed -i -e "s/**steps/{0}/g" run_limits_mc_{1}.jcl'.format(str(args.steps),int(args.mean)))
        os.system('sed -i -e "s/**sig_strength/{0}/g" run_limits_mc_{1}.jcl'.format(m.values["sig_strength"],int(args.mean)))
        os.system('sed -i -e "s/0000/{0}/g" run_limits_mc_{1}.jcl'.format(str(args.ntoys+1),int(args.mean)))
        os.system('sed -i -e "s/**tag/{0}/g" run_limits_mc_{1}.jcl'.format(args.tag,int(args.mean)))
        
        
        os.system('cat run_limits_mc_'+str(int(args.mean))+'.jcl')
        
        
        os.system('condor_submit run_limits_mc_'+str(int(args.mean))+'.jcl')
        print()
        print("submitted limits and reverted the file back")
        print()
        os.system('cat run_limits_mc.jcl')

    df_injs = {
        195:{
            'mean': 250,
            'sigma': 150 
        },
        114:{
            'mean': 250,
            'sigma':200
        }
    }

    if(args.run_bias):
        print("Running Bias")
        for lm in [5.0,2.0,0.0]:
            if lm>1.0: mult=lm-1
            else: mult = lm
            if ss<args.sig_strength: ss=args.sig_strength
            print("Running for lm = ",lm)
            sig_strength=np.ones(500)*m.values['sig_strength']
            sig_error=np.zeros(500)
            # inj_ss = (ss+(lm*ss_sigma))
            inj_ss = df_injs[int(args.mean)]['mean']+lm*df_injs[int(args.mean)]['sigma']
            print(inj_ss, ss, ss_sigma)
            for i in range(500):
                if(i%100==0):print("@",i)
                # print(inj_ss, ss, ss_sigma)
                Y=ys_toys[i] + np.around(inj_ss*gauss_array(x,m.values['mean'],m.values['sigma'],x[1]-x[0]))
                DY=abs((chi2.ppf((1 + 0.681)/2,2*(Y.flatten()+1))/2.)-Y)
                cost = partial(log_likelihood,x=x,y=Y,dy=DY,ym=ym)
                m = Minuit(cost, (params[0], params[1],ss,args.mean,args.sigma), name=("Variance", "length_scale", "sig_strength", "mean", "sigma"))
                m.errordef = 0.5
                m.fixed[0] = True
                m.fixed[1] = True
                m.fixed[3] = True
                m.fixed[4] = True
                m.limits[3] = (args.mean-2*(args.mean_err*args.mean),args.mean+2*(args.mean_err*args.mean))
                m.limits[4] = (args.sigma-2*(args.sigma_err*args.sigma),args.sigma+2*(args.sigma_err*args.sigma))
                # if lm==0:
                #     m.limits[2] = (-3*ss,3*(ss_sigma))
                # else:
                #     m.limits[2] = (ss-(20*lm*ss_sigma),ss+(20*lm*ss_sigma))
                m.migrad()
                m.migrad()
                # print(m)
                sig_strength[i]=m.values['sig_strength']
                mask1 = x<(args.mean-2*args.sigma)
                mask2 = x>(args.mean+2*args.sigma)
                mask = np.logical_or(mask1,mask2)
                # print(mask)
                Xc= x[mask]
                Yc= Y[mask]
                DYc= DY[mask]
                Ycm = ym[mask]
                if(i%100==0):
                    show_fit(params, x, Y, DY, Xc, Yc, DYc,ym,Ycm , m.values['mean'], m.values['sigma'], None,m.values['sig_strength'],'bias_figs/bias3_'+args.tag+'_'+str(int(args.mean))+'_'+str(int(lm))+'_'+str(i)+'3j.pdf')
            bias_array = sig_strength - np.sum(np.around(inj_ss*gauss_array(x,m.values['mean'],m.values['sigma'],x[1]-x[0])))
            bias_array = bias_array/np.std(sig_strength)
            fig = plt.figure(figsize=(8, 6))
            # bins = np.linspace(-5,5,20)
            print(bias_array)
            mean,std=norm.fit(bias_array)
            print("Mean: ",mean," Std: ",std)
            print()
            plt.title("Bias @ "+str(int(lm))+" sigma, "+str(int(args.mean))+" GeV")
            plt.hist(bias_array,bins=20, density=True)
            xmin, xmax = plt.xlim()
            xx = np.linspace(xmin, xmax, 100)
            yy = norm.pdf(xx, mean, std)
            plt.plot(xx, yy,label=r"$\mu$: {0:.2f}," "\n" r"$\sigma$ : {1:.2f}".format(mean,std))
            plt.legend()
            plt.savefig('bias3_'+args.tag+'_'+str(int(args.mean))+'_'+str(int(lm))+'_3j.pdf')






