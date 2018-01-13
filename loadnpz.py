import numpy as np

if __name__ == '__main__':

    save_path = 'model_label_0_param_done.npz'
    model_parameters = np.load(save_path)
    for i in range(model_parameters['cov'].shape[0]):
        np.savetxt("Sigma.txt", model_parameters['cov'][i])