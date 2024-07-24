import time
import datetime

import torch
from torch import nn
import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import logging

logging.set_verbosity_error()
logging.set_verbosity_warning()

from transformers import AdamW
from transformers import get_scheduler
from transformers import get_linear_schedule_with_warmup

from pretrainedModel import load_PretrainedModel
from data_loader import datasetLoader
from data_batcher import dataBatcher
from model import TransformerLSTMClassifier, TransformerWeightedClassifier, TransformerBaselineClassifier
from train_utils import modelTraining
from evaluate import evaluateWithMatrix

def trainer(pretrained_model_type, pooled_model_type, epochs):
    """
        main for training with different architecture
        =============================================================
        args : pretrained_model_type, pooled_model_type <str> 
                epochs <int>
        return : None
    """
    start_time =  time.time()

    if pretrained_model_type == "xlm-roberta":
        _pretrained_model = 'xlm-roberta-base'
    else:
        raise("undefind pretrained model type, that are available xlm-roberta")

    model, tokenizer, config = load_PretrainedModel(_pretrained_model)
    train_dataset, test_dataset, test, encoder = datasetLoader()
    train_dataloader, eval_dataloader = dataBatcher(tokenizer, train_dataset, test_dataset)

    if pooled_model_type == "lstm_pooling":
        modelTransformer = TransformerLSTMClassifier(model, config, 3)
    elif pooled_model_type == "weighted_pooling":
        modelTransformer = TransformerWeightedClassifier(model, config, 3)
    elif pooled_model_type == "baseline":
        modelTransformer = TransformerBaselineClassifier(model, config, 3)
    else:
        raise("unknow model type , there are two type, lstm_pooling and weighted_pooling")

    outputDir = "_".join([pretrained_model_type, pooled_model_type, "ephs_"+str(epochs)])
    outputDir = os.path.join("./models", outputDir)

    print("The model save into {}".format(outputDir))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    modelTransformer.to(device)

    print(" == "* 30)
    print(modelTransformer)
    print(" == "* 30)

    default_save_steps = 1 # by epoch

    # AdamW is an optimizer which is a Adam Optimzier with weight-decay-fix
    optimizer = AdamW(modelTransformer.parameters(),
                    lr = 2e-5, 
                    eps = 1e-8 
                    )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    loss_fct = nn.CrossEntropyLoss()

    modelTraining(outputDir, modelTransformer, tokenizer, train_dataloader, eval_dataloader,
                        optimizer, scheduler, loss_fct, device, epochs)

    evaluateWithMatrix(outputDir, _pretrained_model, modelTransformer, encoder, test, eval_dataloader, device, loss_fct)

    timeDelta = time.time() - start_time
    timeDelta = str(datetime.timedelta(seconds=timeDelta))
    print("Duration : {}".format(timeDelta))