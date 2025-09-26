#!bin/bash
set -e
trap "echo 'Stopping...'; kill 0" SIGINT

datasets=("davis" "kiba")
models=("ESM_GINConvNet" "FRI_GINConvNet" "PDC_GINConvNet" "Vnoc_GINConvNet" "PDC_Vnoc_GINConvNet")
validation_folds=(0 1 2 3 4)
seed=0
num_layers=(1 2 3)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for layer in "${num_layers[@]}"; do
            for fold in "${validation_folds[@]}"; do
                python training.py --seed "$seed" --wandb --dataset "$dataset" --model "$model" --validation_fold "$fold" --num_layers "$layer"&
            done
        done
        wait
    done
done

# for model in "${models[@]}"; do
#     for fold in "${validation_folds[@]}"; do
#         python training.py --seed "$seed" --wandb --dataset "davis" --model "$model" --validation_fold "$fold" --mutation &
#     done
#     wait
# done