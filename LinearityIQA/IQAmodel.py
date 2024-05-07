import Linearity
import KonCept512


def IQAModel(model_name: str, *args, **kwargs):
    if model_name=="Linearity":
        return Linearity.IQAModel(*args, **kwargs)
    elif model_name=="KonCept":
        return KonCept512.KonCept(kwargs.get('arch', 'resnet'))
    raise NameError(f"No {model_name} model.")
