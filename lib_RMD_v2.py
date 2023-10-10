## function calculate the mah dist to mu_c
from scipy import linalg
from scipy.optimize import minimize
from numpy.linalg import norm
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
## def Mah_dist(x1,x2,mu_c,mat_sigam):
int_num_id_class = 8 ## coded form 0 to 7

def get_mu_LDA(df_out, vec_lab, vec_class):
    mat_mu_c = np.empty([len(vec_class),df_out.shape[1]])
    for i in vec_class:
        mat_mu_c[i,:] = np.mean(df_out[vec_lab==i],axis=0)
    return mat_mu_c

def get_mat_sigma_LDA(df_out, vec_lab, vec_class, mat_mu_c):
    int_dim = df_out.shape[1]
    mat_sigma = np.zeros([int_dim, int_dim])
    int_N = df_out.shape[0]
    for i in range(int_N):
        int_cls = np.where(vec_class==vec_lab[i])[0][0]
        mat_sigma = mat_sigma + np.array(df_out.iloc[i,:]-mat_mu_c[int_cls,:]).reshape(-1,1).dot(np.array(df_out.iloc[i,:]-mat_mu_c[int_cls,:]).reshape(1,-1))/(int_N-1)
    return mat_sigma

def get_mu_LDA_CCM(df_out, vec_lab, vec_class):
    mat_mu_c = np.empty([len(vec_class),df_out.shape[1]])
    for i in vec_class:
        mat_mu_c[i,:] = np.median(df_out[vec_lab==i],axis=0)
    return mat_mu_c

def get_mahdist_LDA(df_out, mat_mu, mat_sigma):
    mat_dist = np.zeros(shape=[df_out.shape[0],mat_mu.shape[0]])
    # mat_sigma_inv = np.linalg.inv(mat_sigma)
    for i in range(df_out.shape[0]):
        for j in range(mat_mu.shape[0]):
            mat_dist[i,j] = np.transpose(df_out.iloc[i,:]-mat_mu[j,:])@np.linalg.solve(mat_sigma, df_out.iloc[i,:]-mat_mu[j,:])
    return np.min(mat_dist, axis=1)


def Cov_Shrinkage(df_x,vec_mu, int_iterMAx = 1e3, rho = 1e-4):
    df_x=np.array(df_x)
    n = df_x.shape[0]
    p = df_x.shape[1]
    mat_I = np.identity(p)
    for i in range(n):
        df_x[i,:] = df_x[i,:] - vec_mu
    obj_cov = LedoitWolf().fit(df_x)
    return obj_cov.covariance_

def get_mat_sigma_QDA(df_out, vec_lab, vec_class, mat_mu_c, Shrinkage=False):
    int_dim = df_out.shape[1]
    ll_sigma = []
    for j in vec_class:
        mat_sigma = np.zeros([int_dim, int_dim])
        int_Nj = np.sum(vec_lab==j)
        int_cls = np.where(vec_lab==j)[0]
        for i in int_cls:
            mat_sigma = mat_sigma + np.array(df_out.iloc[i,:]-mat_mu_c[j,:]).reshape(-1,1).dot(np.array(df_out.iloc[i,:]-mat_mu_c[j,:]).reshape(1,-1))/(int_Nj-1)
        ll_sigma.append(mat_sigma)
    if Shrinkage:
        for j in vec_class:
            if min(np.linalg.eigvals(ll_sigma[j]))>0:
                continue
            else:
                int_cls = np.where(vec_lab==j)[0]
                ll_sigma[j] = Cov_Shrinkage(df_out.iloc[int_cls], mat_mu_c[j,:])
    return ll_sigma

def get_mahdist_QDA(df_out, mat_mu, ll_sigma):
    mat_dist = np.zeros(shape=[df_out.shape[0],mat_mu.shape[0]])
    for i in range(df_out.shape[0]):
        for j in range(mat_mu.shape[0]):
            mat_dist[i,j] = np.transpose(df_out.iloc[i,:]-mat_mu[j,:])@np.linalg.solve(ll_sigma[j], df_out.iloc[i,:]-mat_mu[j,:])
    return np.min(mat_dist, axis=1)


def df_mahdist_LDA(tp_df, tp_df_test, vec_lays, num_cls):
    df_out = pd.DataFrame({"dsource":tp_df[0].dsource}) 
    df_out_test = pd.DataFrame({"dsource":tp_df_test[0].dsource})
    ll = 0
    for i in vec_lays:
        df_in = tp_df[ll].copy()
        df_in_test = tp_df_test[ll].copy()
        mu_LDA = get_mu_LDA(df_in.iloc[:,0:i][df_in.dsource=="train_id"],df_in.true_lab[df_in.dsource=="train_id"],range(num_cls))
        sigma_LDA = get_mat_sigma_LDA(df_in.iloc[:,0:i][df_in.dsource=="train_id"],
            df_in.true_lab[df_in.dsource=="train_id"],
            range(num_cls),mu_LDA)
        vec_dist_train = get_mahdist_LDA(df_in.iloc[:,0:i], mu_LDA, sigma_LDA)
        vec_dist_test = get_mahdist_LDA(df_in_test.iloc[:,0:i], mu_LDA, sigma_LDA)
        df_out['dist_l'+str(i)] = vec_dist_train
        df_out_test['dist_l'+str(i)] = vec_dist_test
        del df_in
        ll=ll+1
    return df_out,df_out_test

def df_mahdist_QDA(tp_df, tp_df_test, vec_lays, num_cls, Shrinkage = True):
    df_out = pd.DataFrame({"dsource":tp_df[0].dsource})
    df_out_test = pd.DataFrame({"dsource":tp_df_test[0].dsource})
    ll = 0
    for i in vec_lays:
        df_in = tp_df[ll].copy()
        df_in_test = tp_df_test[ll].copy()
        mu_QDA = get_mu_LDA(df_in.iloc[:,0:i][df_in.dsource=="train_id"],df_in.true_lab[df_in.dsource=="train_id"],range(num_cls))
        sigma_QDA = get_mat_sigma_QDA(df_in.iloc[:,0:i][df_in.dsource=="train_id"],
            df_in.true_lab[df_in.dsource=="train_id"],
            range(num_cls),mu_QDA, Shrinkage=Shrinkage)
        vec_dist_train = get_mahdist_QDA(df_in.iloc[:,0:i], mu_QDA, sigma_QDA)
        vec_dist_test = get_mahdist_QDA(df_in_test.iloc[:,0:i], mu_QDA, sigma_QDA)
        df_out['dist_l'+str(i)] = vec_dist_train
        df_out_test['dist_l'+str(i)] = vec_dist_test
        del df_in, df_in_test
        ll +=1
    return df_out,df_out_test

