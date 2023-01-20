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
    elif name == "resunet_restaining_polyt_dapi_input":
        net = resnets.resunet_restaining_polyt_dapi_input()
    elif name == "resunet_restaining_seqfish_input":
        net = resnets.resunet_restaining_seqfish_input()
    elif name == "resunet_restaining_celltype_input":
        net = resnets.resunet_restaining_celltype_input()
    elif name == "resunet_restaining_color_he_input":
        net = resnets.resunet_restaining_color_he_input()
    else:
        raise NotImplementedError("Could not find network {}.".format(net))
    return net
