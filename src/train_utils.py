import os
import time
import random
import numpy as np
from joblib import dump

import torch

from utils import flat_accuracy, format_time

def save_checkpoint(output_dir, 
                   model, tokenizer, 
                   optimizer, scheduler, 
                   loss, epochs):
    
    """ saving checkpoint for incremental purpose but overwrite every save step"""
    modelDir = os.path.join(output_dir, "checkpoint.pt") 
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'loss': loss,
        }, modelDir)

def modelTrain(model, train_dataloader, optimizer, scheduler, loss_fct, epochs, epoch_i, device):
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss, total_accuracy = 0, 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()
    
    total_all_hidden_states, total_weighted_pooling_embeddings = [], []
    
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 100 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        
        outputs, all_hidden_states, weighted_pooling_embeddings = model(b_input_ids, b_input_mask)
        
        del all_hidden_states
        weighted_pooling_embeddings = weighted_pooling_embeddings.detach().cpu().numpy()
        
#         total_all_hidden_states.append(all_hidden_states)
        total_weighted_pooling_embeddings.append(weighted_pooling_embeddings)
                    
        loss = loss_fct(outputs, b_labels)
        
        #Move logits and labels to CPU
        logits = outputs.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_train_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
        total_accuracy += tmp_train_accuracy
        

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    
    
    # Calculate the average accuarcy over the training data.
    avg_train_accuracy = total_accuracy / len(train_dataloader)
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    
    return model, avg_train_loss, avg_train_accuracy, total_all_hidden_states, total_weighted_pooling_embeddings


def modelEvaluate(model, eval_dataloader, loss_fct, device):
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    predicted_label, predict_proba = [], []
    # total_weighted_pooling_embeddings = []
    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_loss, total_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in eval_dataloader:
        
        # Add batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs, _, _ = model(b_input_ids, b_input_mask)
        

        # total_weighted_pooling_embeddings.append(weighted_pooling_embeddings)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs
        
        loss = loss_fct(logits, b_labels)
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()
    
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        for i_logits in logits:
            predict_proba.append(i_logits)
            predicted_label.append(np.argmax(i_logits))
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
        total_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1
    
    # Calculate the average loss over the training data.
    avg_eval_loss = total_loss / len(eval_dataloader)
    
    
    # Calculate the average accuarcy over the training data.
    avg_eval_accuracy = total_accuracy / len(eval_dataloader)
    
    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(total_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    
    return model, avg_eval_loss, avg_eval_accuracy, predicted_label, predict_proba

def modelTraining(output_dir, model, tokenizer, train_dataloader, eval_dataloader,
                      optimizer, scheduler, loss_fct, device, epochs):
    
    weight_dir = os.path.join(output_dir, "weights")
    
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    args = {}
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    total_train_loss, total_train_accuarcy = [], []
    total_eval_loss, total_eval_accuarcy = [], []
    total_eval_accuracy = []
    save_steps = 0

    # For each epoch...
    for epoch_i in range(0, epochs):

        # Training 
        model, avg_train_loss, avg_train_accuracy, total_all_hidden_states, total_weighted_pooling_embeddings = modelTrain(model, train_dataloader, optimizer, scheduler, loss_fct, epochs, epoch_i, device)
        # Store the loss value for plotting the learning curve.
        total_train_loss.append(avg_train_loss)
        total_train_accuarcy.append(avg_train_accuracy)
        
        # Evaluating
        model, avg_eval_loss, avg_eval_accuracy, _, _ = modelEvaluate(model, eval_dataloader, loss_fct, device)
        # Store the loss value for plotting the learning curve.
        total_eval_loss.append(avg_eval_loss)
        total_eval_accuarcy.append(avg_eval_accuracy)
    
    
        args["epoch"] = epoch_i
        

        save_checkpoint(output_dir, 
                       model, tokenizer, 
                       optimizer, scheduler, 
                       avg_train_loss, epochs)
           
    print("")
    print("Training complete!")
    
    return model