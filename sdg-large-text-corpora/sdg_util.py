"""
A collection of useful variables, functions and classes used to index SDGs

Authors:
* Nick Jelicic (Dialogic)
* Tommy van der Vorst (Dialogic)
* Bijan Ranjbar (MyDataExpert)
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

# INITIALIZATION

# !pip install torch==1.5.0
# !pip install numpy==1.18.4
# !pip install tqdm==4.45.0
# !pip install tokenizers==0.0.2
# !pip install transformers==2.8.0

import re
import os
import pandas as pd
import numpy as np
from transformers import BertPreTrainedModel, BertModel, BertConfig, get_linear_schedule_with_warmup
import tokenizers
import torch.nn as nn
import torch
from tqdm.autonotebook import tqdm


# VARIABLES




SDG_GOALS = [
             "1-Poverty",
             "2-Hunger",
             "3-Health",
             "4-Education",
             "5-Gender",
             "6-Water",
             "7-Energy",
             "8-Work",
             "9-Innovation",
             "10-Inequalities",
             "11-Sustainable Cities",
             "12-Consumption",
             "13-Climate Action",
             "14-Life Below Water",
             "15-Life on Land",
             "16-Institutions",
             "17-Partnerships",
             "unknown"
]

SDG_COLS = ['sdg_%i'% (c+1) for c in range(17)]


def split_to_chunks(text, max_words=400, min_letters=5):
    """
    splits a text string into chunks of length max_words.
    symbols are removed and also very short lines (less than min_letters) are excluded.

    """
    flag_new_line = True
    text_list = []
    this_text = ""
    this_text_length = 0
    for i, t in enumerate(text.split("\n")):

        if len(t) > min_letters:
            t_clean = re.sub('[^0-9a-zA-Z.&!,()+\']', ' ', t)
            t_clean = t_clean.replace("...","").replace(" . . ","")
            this_text_length = this_text_length + len(t_clean.split(" "))
            if this_text_length <= max_words:
                this_text += t_clean + " "
            else:
                text_list.append(this_text.strip()) 
                this_text = t_clean + " "
                this_text_length = 0

    if len(this_text) > 1:
        text_list.append(this_text.strip()) 
    
    return [t for t in text_list if t]


# FUNCTION
def process_text(text):
    """
    turning a text string into a dataframe by adding the chunk order and error status
    """
    text_list = split_to_chunks(text)
    if not text_list:
        text_list = ["ERROR IN READING FILE"]
        error = True
    else:
        error = False
    df = pd.DataFrame({
            "text": text_list,
            "chunk_order": list(range(len(text_list))),
            "error": [error] * len(text_list),
        })
    return df


def process_data(text, tokenizer, max_len):
    """
    process a text string and turns it into tokens needed for the Bert model
    """
    tok_abs = tokenizer.encode(text)
    
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

def process_list_of_text(text_list, model, tokenizer, device):
    """
    processes a list of text and for each one predicts the SDGs
    """

    outs = []

    test_dataset = SDGDataset(
        abstract=text_list, tokenizer=tokenizer)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    tk0 = tqdm(test_data_loader, total=len(test_data_loader))
    with torch.no_grad():
        for bi, d in enumerate(tk0):

                ids = d["ids"]
                token_type_ids = d["token_type_ids"]
                mask = d["mask"]

                ids = ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)

                preds = model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids,
                )

                outs.append(np.round(torch.sigmoid(preds).cpu().detach().numpy(),4))
            
            
    outs = np.vstack(outs)

    outs = list(outs)
    return outs






def smoothen_sdg_values(df_sdg, window_size=5):
    """
    smoothens the SDG values by using the window size
    df_sdg: 
        is pandas data frame with each row corresponding to a text chunk 
        and 17 columns for the 17 sdgs containing the sdg scores coming 
        from the Bert model

    """

    # make the window size smaller in case document is too short
    if df_sdg.shape[0] < 5:
        window_size = df_sdg.shape[0]

    # use rolling window averaging to smoothing the values per SDG
    df_sdg_smooth = df_sdg.rolling(window=window_size).mean()
    
    return df_sdg_smooth


def aggregated_sdg_score(df_sdg_smooth, CONFIDENCE_LEVEL=0.5):
    """
    calculates the sdg scores and top sdg for each document data frame
    df_sdg_smooth: 
        is pandas data frame with each row corresponding to a text chunk 
        and 17 columns for the 17 sdgs containing the smoothened sdg scores

    """
    
    # use the confidence level to make the SDGs binary
    df_sdg_smooth_binary = df_sdg_smooth.copy()
    df_sdg_smooth_binary[df_sdg_smooth_binary < CONFIDENCE_LEVEL] = 0
    df_sdg_smooth_binary[df_sdg_smooth_binary >= CONFIDENCE_LEVEL] = 1
    
    # count number of chunks containing each sdg
    sdg_count = df_sdg_smooth_binary.sum(axis=0)
    
    # count number of rows that have at least one sdg extracted from it
    num_valid_chunks = df_sdg_smooth_binary.max(axis=1).sum() 
    
    # calculate the the score for each sdg
    scores = sdg_count.values / num_valid_chunks
    
    # get the index of the top sdg
    if max(scores)>0:
        top_sdg_index = np.argmax(scores)
    else:
        top_sdg_index = 17

    return scores, top_sdg_index, num_valid_chunks

# BERT CLASSES
class config:
    """
    some basic configurations for the BERT model
    """
    MAX_LEN = 512
    VALID_BATCH_SIZE = 16
    NUM_WORKERS = 16
    BERT_PATH = 'bert-base-uncased'

class SDGModel(BertPreTrainedModel):
    """
    SDGModel used for SDG prediction
    """
    def __init__(self, conf):
        super(SDGModel, self).__init__(conf)
        self.bert = BertModel.from_pretrained(config.BERT_PATH, config=conf)
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

class SDGDataset():
    """
    class for the SDG input
    """
    def __init__(self, abstract, tokenizer):
        self.abstract = abstract
        self.tokenizer = tokenizer
        self.max_len = config.MAX_LEN
    
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
