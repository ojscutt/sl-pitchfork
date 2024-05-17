import numpy as np
import tensorflow as tf

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