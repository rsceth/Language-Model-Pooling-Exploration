import pandas as pd
from joblib import load
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

path = "./project/" 

def datasetLoader():
    """
       data loading that already saved in specific path
    """
    label_correct_dict1 = {
        "LABEL_0": 'neg',
        "LABEL_1": 'neutral',
        "LABEL_2": 'pos'
      }

    label_correct_dict2 = {
        0: 'neg',
        1: 'neutral',
        2: 'pos'
      }


    train = pd.read_csv(path + "data/final/train.csv")
    test = pd.read_csv(path + "data/final/test.csv")

    # train = train[:100]
    # test = test[:10]

    encoder = load(path + "data/final/label_enocder.pkl")
    train.label = encoder.transform(train.label)
    test.label = encoder.transform(test.label)

    train_dataset = Dataset.from_pandas(train) 
    test_dataset = Dataset.from_pandas(test) 
    
    return train_dataset, test_dataset, test, encoder