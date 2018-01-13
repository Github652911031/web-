'''
this file is to train the class GMM model for every class based in the parameters in model_label_raw
'''
from EM_extend import EM_extended

import numpy as np

def load(label):
    #load label_model_parameters
    save_path = 'model_label_raw/' + str(label) + '.npz'
    model_parameters = np.load(save_path)
    return model_parameters

if __name__ == '__main__':
    # label定义训练的类别
    label = 0
    model_parameters = load(label)
    em_ = EM_extended(model_parameters['U'], model_parameters['Sigma'], model_parameters['Pi'], label=label)
    em_.train()