from skopt.callbacks import EarlyStopper
import math

class NanStopper(EarlyStopper):
    """Stop the optimization if the `n_best` minima are within `delta`
    Stop the optimizer if the absolute difference between the `n_best`
    objective values is less than `delta`.
    """
    def __init__(self):
        super(EarlyStopper, self).__init__()

    def _criterion(self, result):
        if math.isnan(result.func_vals[-1]):
            result.func_vals[-1] = 10000
            result.fun = 10000

            # worst is always larger, so no need for abs()
            return True 

        else:
            return None

class ErrorStopper(EarlyStopper):
    """Stop the optimization if the `n_best` minima are within `delta`
    Stop the optimizer if the absolute difference between the `n_best`
    objective values is less than `delta`.
    """
    def __init__(self):
        super(EarlyStopper, self).__init__()

    def _criterion(self, result):
        if ValueError:
            dump(result, './early_stop/result.pkl') 
            return True

        else:
            return None
