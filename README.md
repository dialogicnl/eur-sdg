# EUR-SDG-Mapper Introduction
In 2015 the United Nations has introducted the 17 Sustainable Development Goals (SDGs). The SDGs cover a wide range of sustainability topics such as reducing poverty (SDG-1) and promoting quality education (SDG-4).  

Dialogic Innovatie & Interactie has developed an open-source text classifier (dubbed the EUR-SDG-mapper) for the Erasmus University Rotterdam that can determine the relevance of a (academic) document for each SDG. This repository contains scripts related to the EUR-SDG-Mapper. This repository combines the effort two projects:
*  SDG webservice: webservice/API to make predictions with the model
*  SDG large corpus indexing: Scripts to make predictions over long documents. 

The model is licenced under a GPLv3 licence. For details see LICENCE.

## Model specifications
We fine-tuned BERT on a dataset consisting over 8m (weakly) labeled academic abstracts (sourced from Springer Nature SciGraph and Web of Science) with the use of several sets of keywords, namely: the AURORA SDG Queries, Elsevier and the OSDG Framework. By applying BERT to classify SDG impact we hope that the model, due to the extensive pre-training, will pick up more signals from the texts than just Boolean logic (sometimes single keywords) that form the queries. Validation rounds show that the model is often on par with human judgement. 

## Authors
This repository is the combined effort of multiple parties. Authors:
* Nick Jelicic (Dialogic)
* Tommy van der Vorst (Dialogic)
* Bijan Ranjbar (MyDataExpert)
* Wilfred Mijnhardt (Rotterdam School of Management)
