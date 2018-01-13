#coding:utf8
'''
 SML procedure
for every anno class w, train all pictures(D) which belongs to class w and get the GMM(8) of every picture(EM), use this
  8*D GSM model parameters to get the GMM(64) for this class w(extended EM as a clustering algorithm)

This data_preprocessed is to get GMM model parameterss of all picture which has the semantic label class,
    and use this set of label class to do EM_extened
'''

import os
import numpy as np
from PIL import Image
import scipy.io as sio
import time
from scipy.fftpack import dct
from sklearn.externals import joblib
from sklearn import mixture
from multiprocessing import Pool as pool

def get_data():
    '''
    load the data and convert into ycbcr
    :return:
    train_numpy  the picture used to train in YCbCr
    test_numpy   the picture used to test
    train_annot  the label of the train picture
    test_annot   the label of the test picture
    train_list   the train picture name
    test_list    the test picture name
    word_annot   corresponding word of the label
    '''
    train_annot_file = 'data/corel5k_train_annot.mat'
    train_annot=sio.loadmat(train_annot_file)['annot1']         # 4500, 260
    test_annot_file = 'data/corel5k_test_annot.mat'
    test_annot=sio.loadmat(test_annot_file)['annot2']  # 499, 260

    # load data and covert to numpy
    with open('data/corel5k_words.txt') as f:
        lines = f.readlines()
        word_annot = [line.strip() for line in lines]  # 260
        word_annot = np.array(word_annot)
    with open('data/corel5k_train_list.txt') as f:
        lines = f.readlines()
        train_list = [line.strip() for line in lines]  #4500  train_list[0]:1000/1000
    with open('data/corel5k_test_list.txt') as f:
        lines = f.readlines()
        test_list = [line.strip() for line in lines]   #499  test_list:1000/1001
    train_numpy = []
    for train_path in train_list:
        image = Image.open('data/' + train_path + '.jpeg')
        ycbcr = image.convert('YCbCr')
        ycb_num = np.ndarray((ycbcr.size[1], ycbcr.size[0], 3), 'u1', ycbcr.tobytes())
        train_numpy.append(ycb_num)
    test_numpy = []
    for test_path in test_list:
        image = Image.open('data/' + test_path + '.jpeg')
        ycbcr = image.convert('YCbCr')
        ycb_num = np.ndarray((ycbcr.size[1], ycbcr.size[0], 3), 'u1', ycbcr.tobytes())
        test_numpy.append(ycb_num)

    test_numpy = np.asarray(test_numpy)     # (499, )
    train_numpy = np.asarray(train_numpy)   # (4500, )
    return train_numpy, test_numpy, train_annot, test_annot, train_list, test_list, word_annot


def rolling_window(array, step=2):
    '''
        use a 8*8 window to split the picture to N chunk, return a numpy [N*8*8*3]
        picture_numpy is the numpy file of a picture
        '''
    picture_chunk = []
    row_i = 0
    while not ((row_i + 8) > array.shape[0]):
        col_i = 0
        while not ((col_i + 8) > array.shape[1]):
            chunk = array[row_i:row_i + 8, col_i:col_i + 8, :]
            picture_chunk.append(chunk)
            col_i += step
        row_i += step
    picture_chunk = np.asarray(picture_chunk)
    return picture_chunk


def dct2(picture_n):
    '''
    DCT 2 dimension
    :param picture_n: nx8x8x3
    :return: nx3x8x8
    '''
    picture = np.transpose(picture_n, [0, 3, 1, 2]) # nx3x8x8
    # pic = [dct(dct(picture[i], norm='ortho').T, norm='ortho').T for i in range(picture_n.shape[0])]
    pic = np.transpose(dct(np.transpose(dct(picture, norm='ortho'), [0, 1, 3, 2]), norm='ortho'), [0, 1, 3, 2])
    return np.asarray(pic)

def EM_picture(picture_n):
    '''
    use EM to get the GMM for every picture, return the GMM model the parameters are
    model.weights_(8,), model.means_(8, 192), model.covariances_(8, 192, 192)
    picture_n: [n,3,8,8], n is the number of chunk by rolling
    :return ：[n,30]
    '''
    pict_dct = dct2(picture_n)
    pict_dct_staggered = pict_dct.reshape(pict_dct.shape[0], 3, -1)[:, :, :10].reshape(pict_dct.shape[0], -1)  # to make the YBR staggered
#     X= np.array(np.transpose(output,(1,2,0,3,4)),dtype=np.float16)## 必须强制转成float16 否则内存难以接受.
    model = mixture.GaussianMixture(n_components=8).fit(pict_dct_staggered)  # k = 8, covariances='full'
    return model

def save_model_picture_for_multipro(picture_save_path):
    '''
    to support multiprocessing.pool
    :param picture_save_path[0] picture:
    :param picture_save_path[1] save_path:
    :return:
    '''
    picture = picture_save_path[0]
    save_path = picture_save_path[1]
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + save_path+" start train:")
    picture_n = rolling_window(picture, step=6)
    model_pic = EM_picture(picture_n)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + save_path + " finish.")
    joblib.dump(model_pic, save_path)

def save_model_picture_multiprocessing(train_list, train_numpy):
    '''
    too slow, use pool in multiprocessing
    save GMM model for every picture, for we have split the train and test set, do it split
    :param train_list:
    :param train_numpy:
    :return: every picture model
    '''
    picture_list = []
    save_path_list = []
    start_time = time.time()
    for i, train_path in enumerate(train_list):
        picture = train_numpy[i]
    #     train_path:1000/1000 so we should create directory 1000 first
        save_dir = 'model_picture/' + train_path.split('/')[0]
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = 'model_picture/' + train_path + '.model'
        save_path_list.append(save_path)
        picture_list.append(picture)

    picture_save_path = zip(picture_list, save_path_list)
    p = pool(32)
    p.map(save_model_picture_for_multipro, picture_save_path)
    p.close()
    p.join()
    end_time = time.time()
    print('time for 10 picture is {}'.format(end_time - start_time))

def save_model_label_raw(train_list, train_annot):
    '''
    for a class_label, find all pictures that suits the class_label,
    and load the parameters of this pictures, save to label_index.npz
    for we have split the train and test set, do it split
    :param train_list:
    :param train_annot:
    :return: all pictures model parameters that suits label_index
    '''
    for label_index in range(train_annot.shape[1]):
        weights = []
        means = []
        covs = []
        picture_index_list = np.arange(train_annot.shape[0])[train_annot[:, label_index] > 0]
        for pic_ind in picture_index_list:
            model_path = 'model_picture/' + train_list[pic_ind] + '.model'
            model_pic = joblib.load(model_path)
            weights.append(model_pic.weights_)
            means.append(model_pic.means_)
            covs.append(model_pic.covariances_)
        Pi = np.array(weights)         # D, K
        U = np.array(means)            # D , K, 30
        Sigma = np.array(covs)          # D , K , 30 , 30
        Pi = np.transpose(Pi,[1, 0]) # K,D
        U = np.transpose(U, [1, 0, 2])  # (K,D,30)  30 is the feature_size 8*8*3(rolling window and DCT)
        Sigma = np.transpose(Sigma, [1, 0, 2, 3]) #k,d,30,30
        np.savez_compressed('model_label_raw/' + str(label_index) + '.npz', Pi=Pi, U=U, Sigma=Sigma)
        if label_index % 20 == 0:
            print('has saved {} label model'.format(label_index))


if __name__ == '__main__':
    print('start load data')
    train_numpy, test_numpy, train_annot, test_annot, train_list, test_list, word_annot = get_data()
    print('start save train picture model')

    save_model_picture_multiprocessing(train_list, train_numpy)   #训练每张图片的8个高斯分量，多线程进行
    save_model_label_raw(train_list, train_annot)  #将每类图片的高斯分量存储起来


