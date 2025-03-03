from typing import Optional, Tuple

import torch
from torch import nn
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from transformers import RobertaForSequenceClassification, XLMRobertaXLForSequenceClassification, RobertaModel, XLMRobertaModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class XLMRoBERTaForMultiLabelSequenceClassification(XLMRobertaXLForSequenceClassification):
    def __init__(self, config, coarse_labels_weights=None, fine_labels_weights=None):
        super().__init__(config)
        self.num_coarse_labels = config.num_coarse_labels
        self.num_fine_labels = config.num_fine_labels
        if coarse_labels_weights is not None:
            self.coarse_labels_weights = coarse_labels_weights.to(device)
        else:
            self.coarse_labels_weights = coarse_labels_weights
        if fine_labels_weights is not None:
            self.fine_labels_weights = fine_labels_weights.to(device)
        else:
            self.fine_labels_weights = fine_labels_weights
        self.roberta = XLMRobertaModel(config)
        self.features_layer = FeaturesExtractor(config.hidden_size, config.hidden_dropout_prob)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout else config.hidden_dropout_prob)
        self.coarse_clf = nn.Linear(config.hidden_size, self.num_coarse_labels)
        self.fine_clf = nn.Linear(config.hidden_size + self.num_coarse_labels, self.num_fine_labels)

        self.post_init()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                domains=None,
                coarse_labels=None,
                fine_labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict: Optional[bool] = None
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        roberta_outputs = self.roberta(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       return_dict=return_dict
                                       )
        sequence_outputs = roberta_outputs[0][:, 0, :]
        sequence_features = self.features_layer(sequence_outputs)
        coarse_logits = self.coarse_clf(sequence_features)
        coarse_probs = torch.sigmoid(coarse_logits)
        fine_labels_features = torch.concat((sequence_features, coarse_probs), dim=-1)
        fine_logits = self.fine_clf(fine_labels_features)

        mutual_loss = 0.0
        if coarse_labels is not None:
            # compute loss for coarse labels
            coarse_loss = compute_loss(coarse_logits, coarse_labels, labels_weights=self.coarse_labels_weights)
        else:
            coarse_loss = 0.0
        if fine_labels is not None:
            # compute loss for fine labels
            fine_loss = compute_loss(fine_logits, fine_labels, labels_weights=self.fine_labels_weights)
        else:
            fine_loss = 0.0
        mutual_loss = coarse_loss + fine_loss
        assert return_dict
        return MultiLabelSequenceClassificationOutput(loss=mutual_loss,
                                                      coarse_logits=coarse_logits,
                                                      fine_logits=fine_logits,
                                                      hidden_states=roberta_outputs.hidden_states,
                                                      attentions=roberta_outputs.attentions)

class RoBERTaForMultiLabelSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, coarse_labels_weights=None, fine_labels_weights=None):
        super().__init__(config)
        self.num_coarse_labels = config.num_coarse_labels
        self.num_fine_labels = config.num_fine_labels
        if coarse_labels_weights is not None:
            self.coarse_labels_weights = coarse_labels_weights.to(device)
        else:
            self.coarse_labels_weights = coarse_labels_weights
        if fine_labels_weights is not None:
            self.fine_labels_weights = fine_labels_weights.to(device)
        else:
            self.fine_labels_weights = fine_labels_weights
        self.roberta = RobertaModel(config)
        self.features_layer = FeaturesExtractor(config.hidden_size, config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.classifier_dropout if config.classifier_dropout else config.hidden_dropout_prob)
        self.coarse_clf= nn.Linear(config.hidden_size, self.num_coarse_labels)
        self.fine_clf = nn.Linear(config.hidden_size + self.num_coarse_labels, self.num_fine_labels)

        self.post_init()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                domains=None,
                coarse_labels=None,
                fine_labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict:Optional[bool] = None
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        roberta_outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict
                        )
        sequence_outputs = roberta_outputs[0][:,0,:]
        sequence_features = self.features_layer(sequence_outputs)
        coarse_logits = self.coarse_clf(sequence_features)
        coarse_probs = torch.sigmoid(coarse_logits)
        fine_labels_features = torch.concat((sequence_features, coarse_probs), dim=-1)
        fine_logits = self.fine_clf(fine_labels_features)

        mutual_loss = 0.0
        if coarse_labels is not None:
            # compute loss for coarse labels
            coarse_loss = compute_loss(coarse_logits, coarse_labels, labels_weights=self.coarse_labels_weights)
        else:
            coarse_loss = 0.0
        if fine_labels is not None:
            # compute loss for fine labels
            fine_loss = compute_loss(fine_logits, fine_labels, labels_weights=self.fine_labels_weights)
        else:
            fine_loss = 0.0
        mutual_loss = coarse_loss + fine_loss
        assert return_dict
        return MultiLabelSequenceClassificationOutput(loss=mutual_loss,
                                                      coarse_logits=coarse_logits,
                                                      fine_logits=fine_logits,
                                                      hidden_states=roberta_outputs.hidden_states,
                                                      attentions=roberta_outputs.attentions)

def compute_loss(logits, labels, labels_weights=None):
    if labels_weights is not None:
        criterion = nn.BCEWithLogitsLoss(weight=labels_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()
    return criterion(logits, labels)

class FeaturesExtractor(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.SELU()
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.linear(x)
        x = self.activation(x)
        x = self.output(x)
        return x

@dataclass
class MultiLabelSequenceClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    coarse_logits: torch.FloatTensor = None
    fine_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
