import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import json
from .deepfrier.Predictor import Predictor
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

model_config = 'preprocessing/FRI/trained_models/model_config.json'
ont = 'cc'
emb_layer = 'global_max_pooling1d'

with open(model_config) as json_file:
    params = json.load(json_file)

params = params['cnn']
gcn = params['gcn']
models = params['models']
predictor = Predictor(models[ont], gcn = gcn)

datasets = ['davis', 'kiba']
for dataset in datasets:
    processed_dataset = 'data/' + dataset + '_deepfri.csv'
    if not os.path.isfile(processed_dataset):

        predictor = Predictor(models[ont], gcn=gcn)
        
        df = pd.read_csv('data/' + dataset + '.csv')
        prots = list(df['target_sequence'])
        embeddings = []
        # DeepFRI protein representation
        for i in tqdm(range(0, len(prots)), desc=f"Processing {dataset} proteins"):
            prot = prots[i]
            emb = predictor.predict_embeddings(prot, layer_name = emb_layer)
            embeddings.append(emb)
        print('deepfri embeddings done')
        prots = np.asarray(embeddings)
        print(np.shape(prots))

        df['embeddings'] = list(prots)
        df.to_csv(processed_dataset, index = False)

        print(processed_dataset, ' has been created')
    else:
        print(processed_dataset, ' is already created')