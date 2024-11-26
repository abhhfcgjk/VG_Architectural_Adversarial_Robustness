from models_train import Linearity
from icecream import ic

def IQAModel(*args, **kwargs):
    # ic(model_name)
    # if model_name=="Linearity":
    return Linearity.Linearity(*args, **kwargs)
    # elif model_name=="KonCept":
    #     return KonCept512.KonCept(*args, **kwargs)
    # raise NameError(f"No {model_name} model.")
