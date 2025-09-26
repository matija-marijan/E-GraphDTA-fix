#!bin/bash
set -e
trap "echo 'Stopping...'; kill 0" SIGINT

datasets=("davis" "kiba")
models=("GINConvNet" "ESM_GINConvNet" "FRI_GINConvNet" "PDC_GINConvNet" "Vnoc_GINConvNet" "PDC_Vnoc_GINConvNet" "ESM_GATNet" "GATNet" "GAT_GCN" "GCNNet")
seed=(0 1 2 3 4)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for s in "${seed[@]}"; do
            python training.py --seed "$s" --wandb --dataset "$dataset" --model "$model" &
        done
        wait
    done
done

for model in "${models[@]}"; do
    for s in "${seed[@]}"; do
        python training.py --seed "$s" --wandb --dataset "davis" --model "$model" --mutation &
    done
    wait
done