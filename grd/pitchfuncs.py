import pickle
import numpy as np
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from dynesty import NestedSampler
from dynesty import utils as dyfunc
import scipy
from scipy import constants
from scipy import stats
import astropy.constants

class InversePCA(tf.keras.layers.Layer):
    """
    Inverse PCA layer for tensorflow neural network
    
    Usage:
        - Define dictionary of custom objects containing Inverse PCA
        - Use arguments of PCA mean and components from PCA of output parameters for inverse PCA (found in JSON dict)
        
    Example:

    > f = open("pcann_info.json")
    >
    > data = json.load(f)
    >
    > pca_comps = np.array(data["pca_comps"])
    > pca_mean = np.array(data["pca_mean"])
    > 
    > custom_objects = {"InversePCA": InversePCA(pca_comps, pca_mean)}
    > pcann_model = tf.keras.models.load_model("pcann_name.h5", custom_objects=custom_objects)
    
    """
    
    def __init__(self, pca_comps, pca_mean, **kwargs):
        super(InversePCA, self).__init__()
        self.pca_comps = pca_comps
        self.pca_mean = pca_mean
        
    def call(self, x):
        y = tf.tensordot(x, np.float32(self.pca_comps),1) + np.float32(self.pca_mean)
        return y
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pca_comps': self.pca_comps,
            'pca_mean': self.pca_mean
        })
        return config

class WMSE(tf.keras.losses.Loss):
    """
    Weighted Mean Squared Error Loss Function for tensorflow neural network
    
    Usage:
        - Define list of weights with len = labels
        - Use weights as arguments - no need to square, this is handled in-function
        - Typical usage - defining target precision on outputs for the network to achieve, weights parameters in loss calculation to force network to focus on parameters with unc >> weight.
    
    """
    
    def __init__(self, weights, name = "WMSE",**kwargs):
        super(WMSE, self).__init__()
        self.weights = np.float32(weights)
        
    def call(self, y_true, y_pred):
        loss = ((y_true - y_pred)/(self.weights))**2
        return tf.math.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'weights': self.weights
        })
        return config

def WMSE_metric(y_true, y_pred):
    metric = ((y_true - y_pred)/(weights))**2
    return tf.reduce_mean(metric)


class emulator:
    def __init__(self, emulator_name):
        self.emulator_name = emulator_name
        self.file_path = self.emulator_name
        
        with open(self.file_path+".pkl", 'rb') as fp:
             self.emulator_dict = pickle.load(fp)
            
        self.custom_objects = {"InversePCA": InversePCA(self.emulator_dict['custom_objects']['inverse_pca']['pca_comps'], self.emulator_dict['custom_objects']['inverse_pca']['pca_mean']),"WMSE": WMSE(self.emulator_dict['custom_objects']['WMSE']['weights'])}

        self.model = tf.keras.models.load_model(self.file_path+".h5", custom_objects=self.custom_objects)

        [print(str(key).replace("log_","") + " range: " + "[min = " + str(self.emulator_dict['parameter_ranges'][key]["min"]) + ", max = " + str(self.emulator_dict['parameter_ranges'][key]["max"]) + "]") for key in self.emulator_dict['parameter_ranges'].keys()];

    def predict(self, input_data, n_min=6, n_max=40, verbose=False):
        
        log_inputs_mean = np.array(self.emulator_dict["data_scaling"]["inp_mean"][0])
        
        log_inputs_std = np.array(self.emulator_dict["data_scaling"]["inp_std"][0])

        log_outputs_mean = np.array(self.emulator_dict["data_scaling"]["classical_out_mean"][0] + self.emulator_dict["data_scaling"]["astero_out_mean"][0])
        
        log_outputs_std = np.array(self.emulator_dict["data_scaling"]["classical_out_std"][0] + self.emulator_dict["data_scaling"]["astero_out_std"][0])
        
        log_inputs = np.log10(input_data)
        
        standardised_log_inputs = (log_inputs - log_inputs_mean)/log_inputs_std

        standardised_log_outputs = self.model(standardised_log_inputs)

        standardised_log_outputs = np.concatenate((np.array(standardised_log_outputs[0]),np.array(standardised_log_outputs[1])), axis=1)

        log_outputs = (standardised_log_outputs*log_outputs_std) + log_outputs_mean

        outputs = 10**log_outputs

        outputs[:,2] = log_outputs[:,2] ##we want star_feh in dex

        # def calc_Teff(luminosity, radius):
        #     return np.array(((luminosity.values*astropy.constants.L_sun) / (4*np.pi*constants.sigma*((radius.values*astropy.constants.R_sun)**2)))**0.25)


        teff = np.array(((outputs[:,1]*astropy.constants.L_sun) / (4*np.pi*constants.sigma*((outputs[:,0]*astropy.constants.R_sun)**2)))**0.25)
        
        outputs[:,0] = teff
        
        outputs = np.concatenate((np.array(outputs[:,:3]), np.array(outputs[:,n_min-3:n_max-2])), axis=1)

        return outputs