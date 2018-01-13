# coding:utf-8
import traceback
import numpy as np
from multiprocessing import Pool
import scipy
from scipy.stats import multivariate_normal
from sklearn.externals import joblib
import time

import logging
import logging.handlers


# return GSM model
def gauss(X, U, Sigma):
    '''
    多变量高斯分布的概率密度函数
    Log of the probability density function
    '''
    r = multivariate_normal.logpdf(X, mean=U, cov=Sigma)

    return r


def rel_error(x, y):
    '''returns relative error '''
    return np.max(np.abs(x - y) / (np.maximum(1e2, np.abs(x) + np.abs(y))))


def gauss_p(x):
    return gauss(x[0], x[1], x[2])


def h_function(x):
    '''
    h_function calculate what in exp()
    this h_function is at least ten times faster than the following h_function
    because trace(np.dot(a, b)) = sum(a * b.T)  when a,b are [2,2] matrix , this is very obvious
    in the case, the broadcast is used in np.dot, and we get(k,d,192,192), so we should sum(axis=2).sum(axis=2), return(k,d)
    :param x:list
    x[0] (k, d, 192, 192)  sigma_j_k
    x[1] (192, 192)         sigma_c_m
    :return: return all k*d output of this m
    (k*d)
    '''
    # np.linalg.inv()：inverse the matrix
    return -0.5 * (x[0] * np.linalg.inv(x[1]).T).sum(axis=2).sum(axis=2)




class EM_extended:
    '''
    extened EM model, we have K*D GSM models, and we want to use it to cluster M GSM mdels

    '''

    def __init__(self, U, Sigma, Pi, label, M=64, max_iteration=100, toi=1e-1):
        '''
        初始化模型, 主要参数包括:

        U =>self.iU: K*D*30 类相关图片模型的均值
        Sigma=>self.iSigma: K*D*30*30 类相关图片模型的协方差矩阵
        Pi=>self.iPi: K*D 权值

        类本身的属性
        self.U : M*30 类相关图片模型的均值
        self.Sigma: M*30*30 类相关图片模型的协方差矩阵
        self.Pi: M 权值

        toi: 迭代终止误差: delta=abs(new_param-old_param). toi<sum(delta)/Reg+max(delta)时终止迭代,
        也就是当参数经过迭代之后不再发生显著变化时,停止迭代.
        '''
        LOG_FILE = 'train' + str(label) + '.log'
        handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=102400 * 1024, backupCount=5)  # 实例化handler
        fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
        formatter = logging.Formatter(fmt)  # 实例化formatter
        handler.setFormatter(formatter)  # 为handler添加formatter
        self.logger = logging.getLogger('tst' + str(label))  # 获取名为tst的logger
        self.logger.addHandler(handler)  # 为logger添加handler
        self.logger.setLevel(logging.DEBUG)
        self.logger.info('----------------------welcome to the new age : %s----------------' % label)
        self.K, self.D, self.feature_size = U.shape

        # init self.U, self.Sigma, self.Pi, and the extened EM is to return self.U, self.Sigma, self.Pi
        self.M = M  # 64
        # self.U = np.random.random([M, self.feature_size]) * 0.0001 + np.mean(U, axis=0).mean(axis=0)  ### 64*192
        self.U = np.random.random([M, self.feature_size]) * 0.0001 + self.init_U(U, M)
        self.Sigma = np.zeros([self.M, self.feature_size, self.feature_size]) + np.eye(self.feature_size, self.feature_size)  # 64*192*192
        tmp = np.random.random(M)
        self.Pi = tmp / np.sum(tmp)  # M

        self.errors = []

        self.logger.info(self.Pi)
        # i-prefix indicates the original model parameters
        self.iU = U  # K,D,30
        self.iSigma = Sigma  # K,D,30,30
        self.iPi = Pi  # K,D
        self.max_iteration = max_iteration
        self.toi = toi
        self.h = np.zeros([self.K, self.D, self.M])  ##K,D,M
        self.label = label

    def init_U(self, U, M):
        U_max = np.amax(U.reshape(-1, self.feature_size), axis=0)
        U_min = np.amin(U.reshape(-1, self.feature_size), axis=0)
        U_delta = (U_max - U_min)/M
        U_init = np.zeros((M, self.feature_size))
        for i in range(M):
            U_init[i] = U_min + i*U_delta
        return U_init

    def train(self):
        M, K, D = self.M, self.K, self.D
        #        p=Pool(1)
        iteration = 0  # iteration num
        while (iteration < self.max_iteration):
            try:
                self.logger.info('This is iteration {}'.format(iteration))
                starttime = time.time()
                iteration += 1

                gauss_var = (zip([self.iU.reshape(K * D, self.feature_size)] * self.U.shape[0], self.U, self.Sigma))
                # # gasuss_var: ( (64,k*d,30), (64,30), (64,30,30))

                h_g = np.transpose(np.array(list(map(gauss_p, gauss_var))).reshape(M, K, D), [1, 2, 0]).reshape(K, D, M)  # K,D,M
                # must add list(map(..,..)) for map will return a map function not a list
                # h_g is the G(u_jk.u_cm,sigma_cm)  u_jk is like the input x, and h_g is like the output y


                h_e = np.transpose(np.array(list(map(h_function, list(zip([self.iSigma] * self.Sigma.shape[0], self.Sigma))))).reshape(M, K, D), [1, 2, 0])
                # h_function() return k*d, map to (m,k*d) reshape to(m,k,d)  transpose to (k,d,m)
                # print(h_e.shape)

                h_g = h_g.astype(np.float128)
                in_exp = (1 * (h_g + h_e) * self.iPi.reshape(K, D, 1)) + np.log(self.Pi)


                fenzi = np.exp(in_exp)
                # at first, we use log to calculate h_e and h_g, to get the fenzi, we should add exp. fenzi((k,d,m))

                fenmu = (np.sum(fenzi, axis=2).reshape(K, D, 1))

                # sum in M  (k,d,1)
                eps = 1e-320  # a very small positive number to avoid the fenmu=0, 1e-323 is the min in python
                if np.sum(fenmu == 0) == 0:
                    # fenmu == 0 return True or False for every (k,d), fenmu cannot be 0, if all fenmu are not 0, we dono need eps
                    eps = 0
                self.h = (fenzi + eps / M) / (fenmu + eps)  # (k,d, m)
                # self.h (k,d,m)
                # print('self.h', self.h)
                self.logger.info('self.h')
                self.logger.info(self.h.shape)
                self.logger.info(self.h)
                # -----------------------------M STEP------------------------------------

                pre_param = (self.U, self.Sigma, self.Pi)
                self.Pi = self.h.sum(axis=0).sum(axis=0) / (D * K)

                w = self.h * (self.iPi.reshape(K, D, 1))  # (k,d,m) * (k,d,1)  broadcast
                w /= (w.sum(axis=0).sum(axis=0))  # (k,d,m)
                for m in range(M):
                    # for the specific m, calculate all jk
                    u_delta = self.iU.reshape(K, D, self.feature_size, 1) - self.U[m].reshape(1, 1, self.feature_size, 1)
                    # (k,d,192,1) - (1,1,192,1)  broadcast to all k d    (k,d,192,1)
                    u_delta_T = u_delta.reshape(K, D, 1, self.feature_size)
                    # u_delta(k, d, 192, 1) * u_delta_T(k,d,1,192)   broadcast   (k, d, 192, 192)
                    sigma_tmp_m = w[:, :, m].reshape(K, D, 1, 1) * (self.iSigma + (u_delta) * (u_delta_T))
                    self.Sigma[m] = sigma_tmp_m.sum(axis=0).sum(axis=0)  # sum in k d

                self.U = (w.reshape(K, D, M, 1) * (self.iU.reshape(K, D, 1, self.feature_size))).sum(axis=0).sum(axis=0)  # 更新Sigma
                # self.U = np.dot(w.reshape(K, D, M, 1) , self.iU.reshape(K, D, 1, self.feature_size)).sum(axis=0).sum(axis=0)  # 更新Sigma
                # if use np.dot(), the memory fails, the front code is same as the np.dot for the broadcast in numpy
                # self.U(M, 30)
                # ----------------------------log and delta compute-------------------------
                self.logger.info('self.Sigma')
                self.logger.info(self.Sigma)
                self.logger.info('self.Pi')
                self.logger.info(self.Pi)
                self.logger.info('self.U')
                self.logger.info(self.U)
                endtime = time.time()
                self.logger.info('time is {}'.format(endtime - starttime))
                self.logger.info('-----error----------')
                error1 = np.abs(self.U - pre_param[0])
                e1 = np.max(error1)
                e2 = np.sum(error1) / 192
                self.logger.info('U error : max_U_error{0} ave_U_error{1}'.format(e1, e2))

                self.errors.append((e1, e2))
                print('max U error {}'.format(e1))
                np.savez_compressed('model_label_{}_param_done_{}'.format(self.label, iteration), U=self.U, cov=self.Sigma, pi=self.Pi)

            except Exception as e:
                traceback.print_exc()  # print the Exception information in details
                (self.U, self.Sigma,
                 self.Pi) = pre_param  # if Exception raised, ignore this update and save the model param
                np.savez_compressed('model_label_{}_param_exception'.format(self.label), U=self.U, cov=self.Sigma, pi=self.Pi)
                return self

        # 保存模型参数
        np.savez_compressed('model_label_{}_param_done'.format(self.label), U=self.U, cov=self.Sigma, pi=self.Pi)
        return self
