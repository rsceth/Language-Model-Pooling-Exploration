import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoConfig, AutoTokenizer
)

def load_PretrainedModel(_pretrained_model):
    """
        load pretrained model using model type,
        if it doesn't in local folder, it will download
        ======================================================
        args : _pretrained_model <str>
        return : model <transformer model>, 
                tokenizer <transformer tokenizer>, 
                config <transformer config>
    """
    config = AutoConfig.from_pretrained(_pretrained_model)
    config.update({'output_hidden_states':True})
    model = AutoModel.from_pretrained(_pretrained_model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(_pretrained_model)
    return model, tokenizer, config