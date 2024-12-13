from typing import Callable, Tuple, List
import numpy as np
import inspect


class Transform():
    def __init__(self, fn:Callable):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)
    
    def __repr__(self):
        params = ", ".join(f"{key}={value}" for key, value in vars(self).items() if key != "fn")
        return f"{self.__class__.__name__}({params})"
    

    # def __repr__(self):
    #     # Dynamically extract parameters from the __init__ method
    #     init_signature = inspect.signature(self.__init__)
    #     params = {
    #         key: getattr(self, key, None)
    #         for key in init_signature.parameters
    #         if key != 'self'
    #     }
    #     params_repr = ", ".join(f"{key}={value}" for key, value in params.items())
    #     return f"{self.__class__.__name__}({params_repr})"

class TransZRotation(Transform):
    
    def fn(self, x):
        r = np.random.uniform(*self.limits)
        rot_matrix = np.array([[np.cos(r), -np.sin(r), 0],
                                [np.sin(r),  np.cos(r), 0],
                                [0,          0,         1]])
        return x @ rot_matrix
    
    def __init__(self, limits:Tuple = (0, 2*np.pi)):
        self.limits = limits
            
        super().__init__(self.fn)

class TransCenterXyz(Transform):
    def fn(self, x:np.ndarray):
        return x - x.mean(axis=0)

    def __init__(self):
        super().__init__(self.fn)

class TransScaling(Transform):
    def fn(self,x):
        scale_factor = np.random.uniform(*self.limits)
        scale_matrix = np.diag([scale_factor, scale_factor, scale_factor])  # Uniform scaling
        return x @ scale_matrix
    
    def __init__(self, limits: Tuple = (0.85, 1.15)):
        self.limits = limits
        
        super().__init__(self.fn)

class TransGaussianNoise(Transform):
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean, self.std = mean, std
        def fn(x:np.ndarray):
            noise = np.random.normal(loc=mean, scale=std, size=x.shape)
            return x + noise

        super().__init__(fn)

class TransGammaCorrection(Transform):
    
    def fn(self, x):
        gamma = np.random.uniform(*self.limits)
        return np.clip(x ** gamma, 0, 1) # TODO: Check order of normalization and augmentation !!!
    
    def __init__(self, limits: Tuple[float, float] = (0.8, 1.2)):
        self.limits = limits
        super().__init__(self.fn)

class TransSignalScaling(Transform):
    def fn(self, x:np.ndarray):
        # waveform shape is neibors, 32
        scale_factors = np.random.uniform(*self.limits, size=(x.shape[0],1))
        return x * scale_factors
    
    def __init__(self,limits:Tuple[float,float] = (0.95,1.05)):
        self.limits = limits
        
        super().__init__(self.fn)
        
class TransFeatureDropout(Transform):
    def fn(self,x):
        mask = np.random.binomial(1, 1 - self.dropout_prob, size=x.shape[1])
        return x * mask  
    def __init__(self, dropout_prob: float = 0.2):
        self.dropout_prob = dropout_prob
        
        super().__init__(self.fn)

class TransStandardize(Transform):
    def fn(self, x:np.ndarray):
        x -= self.mean
        x /= self.std
        return x
    def __init__(self, mean, std):
        self.mean, self.std = mean, std
        super().__init__(self.fn)

class TransformsList():
    def __init__(self,
                 transforms:List[Transform]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    def __repr__(self):
        transforms_repr = ",\n  ".join(repr(t) for t in self.transforms)
        return f"TransformsList([\n  {transforms_repr}\n])"






