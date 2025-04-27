from .DRPCANet import *

def get_model(name, net=None):
    if name == 'Drpcanet':
        net = DRPCANet(stage_num=6)
    else:
        raise NotImplementedError

    return net

