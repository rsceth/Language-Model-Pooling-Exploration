import torch
from torch import nn

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, num_labels, embedding_size1, embedding_size2):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dense = nn.Linear(embedding_size1, embedding_size2)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(embedding_size2, num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LSTMPooling(nn.Module):
    """ LSTM layer for roberta representation  """
    def __init__(self, config):
        super(LSTMPooling, self).__init__()
        self.config = config
        self.embedding_size = 256
        self.lstm = nn.LSTM(self.config.hidden_size, self.embedding_size, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, all_hidden_states):
        
        ## forward
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                    for layer_i in range(1, self.config.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.config.num_hidden_layers, self.config.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out

class TransformerLSTMClassifier(nn.Module):
    """ Roberta + LSTM Pooling + linear classifier"""
    def __init__(self, model, config, num_labels):
        super().__init__()
        self.roberta = model
        self.pooler = LSTMPooling(config)
        self.classifier = RobertaClassificationHead(config, num_labels, self.pooler.embedding_size * 2,
                                                            self.pooler.embedding_size)
        
    def forward(self, b_input_ids, b_input_mask, **kwargs):
        outputs = self.roberta(b_input_ids, b_input_mask)
        all_hidden_states = torch.stack(outputs[2])
        weighted_pooling_embeddings = self.pooler(all_hidden_states)
        logits = self.classifier(weighted_pooling_embeddings)
        return logits, all_hidden_states, weighted_pooling_embeddings


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class TransformerWeightedClassifier(nn.Module):
    def __init__(self, model, config, num_labels):
        super().__init__()
        self.roberta = model
        self.pooler = WeightedLayerPooling(config.num_hidden_layers)
        self.classifier = RobertaClassificationHead(config, num_labels, config.hidden_size, config.hidden_size)
        
    def forward(self, b_input_ids, b_input_mask, **kwargs):
        outputs = self.roberta(b_input_ids, b_input_mask)
        all_hidden_states = torch.stack(outputs[2])
        weighted_pooling_embeddings = self.pooler(all_hidden_states)
        weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]
        logits = self.classifier(weighted_pooling_embeddings)
        return logits, all_hidden_states, weighted_pooling_embeddings

class RobertaClassificationHeadBase(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        weight = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(weight)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x, weight

class TransformerBaselineClassifier(nn.Module):
    def __init__(self, model, config, num_labels):
        super().__init__()
        self.config = config
        self.roberta = model
        self.classifier = RobertaClassificationHeadBase(config, num_labels)

    def forward(self, b_input_ids, b_input_mask, **kwargs):
        outputs = self.roberta(b_input_ids, b_input_mask)
        sequence_output = outputs[0]
        logits, weight = self.classifier(sequence_output)

        return logits, sequence_output, weight