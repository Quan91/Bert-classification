import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)  # (768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, o2 = self.bert(input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          token_type_ids=token_type_ids)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output