import torch
import os

def save_model(encoder, decoder, identifier):
    """_summary_

    Args:
        encoder (_type_): _description_
        decoder (_type_): _description_
        identifier (str): suffix of the filename should not include "encoder" or "decoder" and does not inclue file extension, .pth
    """
    if not os.path.isdir("./saved_models"):
        os.makedirs("saved_models")
    
    encpath = os.path.join("saved_models", f"encoder_{identifier}.pth")
    decpath = os.path.join("saved_models", f"decoder_{identifier}.pth")

    torch.save(encoder.state_dict(), encpath)
    torch.save(decoder.state_dict(), decpath)

def load_model(encoder, decoder, identifier):
    """_summary_

    Args:
        encoder (_type_): _description_
        decoder (_type_): _description_
        identifier (str): suffix of the filename should not include "encoder" or "decoder" and does not inclue file extension, .pth
    """
    encpath = os.path.join("saved_models", f"encoder_{identifier}.pth")
    decpath = os.path.join("saved_models", f"decoder_{identifier}.pth")

    encoder.load_state_dict(torch.load(encpath))
    decoder.load_state_dict(torch.load(decpath))
