import numpy as np

def make_bar_dataset(
  n_samples: int = 1200,
  size: int = 8,
  noise: float = 0.20,
  seed: int=0,  
):
    """
    Create small grayscale images.

    Class 0: a vertical two-pixel-wide bar.
    Class 1: a horizontal two-pixel-wide bar.

    Returns:
        X_train: (N_train, 1, size, size)
        y_train: (N_train,)
        X_test:  (N_test, 1, size, size)
        y_test:  (N_test,)
    """
    
    rng = np.random.default_rng(seed)
    
    X = np.zeroes(
        (n_samples, 1, size, size),
        dtype = np.float32,
    )
    
    y = rng.integers(
        low=0,
        high=2,
        size=n_samples,
        dtype=np.int64
    )
    
    for n, label in enumerate(y):
        image = rng.normal(
            loc=0.0,
            scale = noise,
            size=(size,size),
        ).astype(np.float32)
        
        if label==0:
            column = rng.integers(2, size -2)
            image[:, column:column + 2] += 1.0
            
        else:
            row = rng.integers(2, size -2)
            image[row:row+2, :]+= 1.0
            
        X[n, 0] = np.clip(image, 0.0, 1.0)
        
    indices = rng.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    split = int(0.8*n_samples)
    
    X_train = X[:split]
    y_train = y[:split]
    
    X_test = X[split:]
    y_test = y[split:]
    
    return X_train, y_train, X_test, y_test

class Conv2D:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int=0,
        seed: int=0      
    ):
        rng = np.random.default_rng(seed)
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size