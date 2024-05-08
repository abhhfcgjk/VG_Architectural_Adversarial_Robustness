import LinearityIQA.Linearity as Linearity
import LinearityIQA.KonCept512 as KonCept512

from icecream import ic
def IQAModel(model_name: str, *args, **kwargs):
    ic(model_name)
    if model_name=="Linearity":
        return Linearity.IQAModel(*args, **kwargs)
    elif model_name=="KonCept":
        return KonCept512.KonCept(kwargs.get('arch', 'resnet'))
    raise NameError(f"No {model_name} model.")
