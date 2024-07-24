from sklearn.metrics import classification_report, confusion_matrix
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import torch

from pretrainedModel import load_PretrainedModel
from model import TransformerLSTMClassifier
from train_utils import modelEvaluate

def classification_report_csv(report, evaluatedDir):
    """
        saving sklearn classification report into csv file
        =====================================================
        args: report, evaluatedDir <str>
        return : None
    """
    report_data = []
    lines = report.split('\n')
    for line in lines:
        row = {}
        row_data = line.split('      ')
        row_data = [x for x in row_data if len(x) > 0]

        try:
            row['class'] = row_data[0].replace(" ", "")
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
        except:
            pass
        if len(row) > 0:
            report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(os.path.join(evaluatedDir, 'classification_report.csv'), index = False)

def evaluateWithMatrix(outputDir, _pretrained_model, modelTransformer, encoder, test, eval_dataloader, device, loss_fct):
    """
        Evaluate with accuarcy, f1 score, precision and recall
        =====================================================
        args: outputDir, modelDir <str>
        return : None
    """
    modelDir = os.path.join(outputDir, "checkpoint.pt")
    checkpoint = torch.load(modelDir)

    model, tokenizer, config = load_PretrainedModel(_pretrained_model)
    model = modelTransformer
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model, avg_eval_loss, avg_eval_accuracy, predicted_label, predict_proba = modelEvaluate(model, eval_dataloader, loss_fct, device)

    predicted_label = encoder.inverse_transform(predicted_label)
    actual_label = encoder.inverse_transform(test.label.to_list())

    report = classification_report(predicted_label, actual_label)
    cf_matrix = confusion_matrix(predicted_label, actual_label)

    label_name = encoder.classes_.tolist()
    cm_array_df = pd.DataFrame(cf_matrix, index=label_name, columns=label_name)
    sns.heatmap(cm_array_df, annot=True, cmap='Blues', fmt='g')

    evaluatedDir = os.path.join(outputDir, "evaluate")
    if not os.path.exists(evaluatedDir):
        os.makedirs(evaluatedDir)


    plt.savefig(os.path.join(evaluatedDir, 'confusionMatrix.png'), dpi=400)
    classification_report_csv(report, evaluatedDir)

    
    

