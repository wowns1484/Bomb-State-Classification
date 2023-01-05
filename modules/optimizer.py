from torch.optim import *
import json

_optimizer_entrypoints = {
    "adam": Adam,
    "adamW": AdamW,
    "sgd": SGD,
}

def optimizer_entrypoint(scheduler_name):
    return _optimizer_entrypoints[scheduler_name]

def is_optimizer(optimizer_name):
    return optimizer_name in _optimizer_entrypoints

def create_optimizer(parameter, optimizer_name):
    if is_optimizer(optimizer_name):
        with open('/opt/ml/meca/modules/configs/optimizer.json') as f:
            optimizer_args = json.load(f)[optimizer_name]
        
        create_fn = optimizer_entrypoint(optimizer_name)
        optimizer = create_fn(parameter, **optimizer_args)
    else:
        raise RuntimeError('Unknown optimizer (%s)' % optimizer_name)
    return optimizer