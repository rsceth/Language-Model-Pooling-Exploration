import torch
from model import TransformerLSTMClassifier
from pretrainedModel import load_PretrainedModel


class Inference(onject):
    """ for each sentence predict using specific model """
    def __init__(self, modelDir):
        model, self.tokenizer, config = load_PretrainedModel(_pretrained_model)
        self.model = TransformerLSTMClassifier(model, config, 3)
        checkpoint = torch.load(modelDir)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def inferenceRealTime(self, text):
        """
             main for model 
             ===========================================
             args : text <str>
             return : predicted_label <str>
        """
        inputs = self.okenizer(text, truncation=True, padding=True)
        b_input_ids = torch.tensor(np.array([inputs['input_ids']]), device = device)
        b_input_mask = torch.tensor(np.array([inputs['attention_mask']]), device = device)
        outputs = self.model(b_input_ids, b_input_mask)
        outputs = outputs.detach().cpu().numpy().tolist()
        outputs = np.argmax(outputs[0])
        predicted_label = encoder.inverse_transform([outputs])
        return predicted_label