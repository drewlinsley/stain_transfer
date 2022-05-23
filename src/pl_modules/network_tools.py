from torch import nn

from src.pl_modules import resnets


def get_network(name):
    """Wrapper for selecting networks."""
    if name == "resunet":
        net = resnets.resunet()
    elif name == "resunet_control":
        net = resnets.resunet_control()
    elif name == "resunet_restaining":
        net = resnets.resunet_restaining()
    else:
        raise NotImplementedError("Could not find network {}.".format(net))
    return net
