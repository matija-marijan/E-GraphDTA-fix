import os
import pandas as pd
import numpy as np
import esm
from rdkit import Chem
import networkx as nx
from utils import *
import argparse
from tqdm import tqdm

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
if torch.cuda.is_available():
    model = model.cuda()
batch_converter = alphabet.get_batch_converter()
embeddings = []

datasets = ['davis', 'kiba']
for dataset in datasets:

    processed_dataset = 'data/' + dataset + '_esm.csv'

    if not os.path.isfile(processed_dataset):
        df = pd.read_csv('data/' + dataset + '.csv')

        protein_list = list(df['target_sequence'])
        batch_size = 4 if dataset == 'davis' else 1
        labels = []
        for i in range(0, len(protein_list)):
            labels.append('protein' + str(i + 1))
        protein_list = list(zip(labels, protein_list))

        for i in tqdm(range(0, len(protein_list), batch_size), desc=f"Processing {dataset} proteins"):

            batch_prots = protein_list[i : i + batch_size]

            batch_labels, batch_strs, batch_tokens = batch_converter(batch_prots)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                if torch.cuda.is_available():
                    results = model(batch_tokens.cuda(), repr_layers=[6])
                else:
                    results = model(batch_tokens, repr_layers = [6])
            token_representations = results["representations"][6]

            sequence_representations = []
            for j, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[j, 1 : tokens_len - 1].mean(0).cpu())

            for j in range(0, len(sequence_representations)):
                embeddings.append(sequence_representations[j])

        embeddings = np.asarray(embeddings)
        df['embeddings'] = list(embeddings)
        df.to_csv(processed_dataset, index = False)

        print(processed_dataset, ' has been created')
    else:
        print(processed_dataset, ' is already created')