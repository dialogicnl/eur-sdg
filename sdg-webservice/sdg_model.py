
"""
SDG model specifications

Authors:
* Nick Jelicic (Dialogic)
* Tommy van der Vorst (Dialogic)
* Wilfred Mijnhardt (Rotterdam School of Management)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
import tokenizers
import config

class SDGconfig:
    MAX_LEN = 512
    VALID_BATCH_SIZE = 16
    MODEL_PATH = config.MODEL_PATH,
    BERT_PATH = 'bert-base-uncased'
    TOKENIZER = tokenizers.BertWordPieceTokenizer(
        config.VOCAB_PATH,
        lowercase=True
    )
def process_data(abstract,tokenizer, max_len):
    tok_abs = tokenizer.encode(abstract)

    input_ids_orig = tok_abs.ids


    token_type_ids = [1] * (len(input_ids_orig))
    mask = [1] * len(token_type_ids)


    padding_length = max_len - len(input_ids_orig)
    if padding_length > 0:
        input_ids_orig = input_ids_orig + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)


    if  padding_length < 0:
        input_ids_orig = input_ids_orig[:512]
        mask = mask[:512]
        token_type_ids = token_type_ids[:512]


    return {
        'ids': input_ids_orig,
        'mask': mask,
        'token_type_ids':token_type_ids


    }

class SDGDataset:
    def __init__(self, abstract):
        self.abstract = abstract
        self.tokenizer = SDGconfig.TOKENIZER
        self.max_len = SDGconfig.MAX_LEN

    def __len__(self):
        return len(self.abstract)

    def __getitem__(self, item):
        data = process_data(
            self.abstract[item],
            self.tokenizer,
            self.max_len
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long)
        }


class SDGModel(BertPreTrainedModel):
    def __init__(self, conf):
        super(SDGModel, self).__init__(conf)
        self.bert = BertModel.from_pretrained(SDGconfig.BERT_PATH, config=conf)
        self.drop_out = nn.Dropout(0.1)


        self.nb_features = self.bert.pooler.dense.out_features

        self.pooler = nn.Sequential(
            nn.Linear(self.nb_features * 2, self.nb_features),
            nn.Tanh(),
        )

        self.logit = nn.Linear(self.nb_features, 17)

        torch.nn.init.normal_(self.logit.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        last_hidden_state ,pooler_output , hidden_states = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        out = torch.cat((hidden_states[-1][:, 0], hidden_states[-2][:, 0]), dim=-1)

        out = self.drop_out(out)
        out = self.pooler(out)

        logits = self.logit(out)


        return logits
