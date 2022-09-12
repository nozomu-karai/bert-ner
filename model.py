import torch.nn as nn
from transformers import AutoModel

class NERModel(nn.Module):
    def __init__(self, model, ne_dict):
        super().__init__()

        self.ne_dict = ne_dict
        self.bert = AutoModel.from_pretrained(model)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, len(ne_dict))
        self.relu = nn.ReLU()

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask, token_type_ids, valid, label=None):
        output = self.bert(input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
        )
        last_hidden = output['last_hidden_state']

        x = self.relu(self.fc1(last_hidden))
        output = self.relu(self.fc2(x))

        if label is not None:
            output = output * valid.unsqueeze(2)
            _, _, t = output.shape
            loss = self.loss_fct(output.view(-1, t), label.view(-1))
            
            return loss, output

        return output

