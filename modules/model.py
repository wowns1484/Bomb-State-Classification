from .models.efficientnet import EfficientNet
from .models.vgg import VGG

_model_entrypoints = {
    'efficientnet': EfficientNet,
    'vgg': VGG,
}

def model_entrypoint(model_name):
    return _model_entrypoints[model_name]

def is_exist_model(model_name):
    return model_name in _model_entrypoints

def create_model(model_name, **kwargs):
    if is_exist_model(model_name):
        create_model = model_entrypoint(model_name)
        model = create_model(**kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)
    
    return model