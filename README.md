# bert-ssl-moleculenet

## Run Supervised Training with Augmentation

```python
from tqdm import tqdm

SAMPLES_PER_CLASS = [50, 100, 150, 200, 250]
N_AUGMENT = [2, 4, 8, 16, 32]
datasets = ['bace', 'bbbp']
out_file = 'eval_result_supervised_augment.csv'
N_TRIALS = 20
EPOCHS = 20

for dataset in datasets:
    for SAMPLE in SAMPLES_PER_CLASS:
        for n_augment in N_AUGMENT:
            for i in tqdm(range(N_TRIALS)):
                !python pseudo_label/main.py --dataset-name={dataset} --epochs={EPOCHS} \
                --batch-size=16 --model-name-or-path=shahrukhx01/muv2x-simcse-smole-bert \
                --samples-per-class={SAMPLE} --eval-after={EPOCHS} --train-log=0 --train-ssl=0 \
                --out-file={out_file} --n-augment={n_augment}
                !cat {out_file}
```

## Run Pseudo Label Training

```python
from tqdm import tqdm

SAMPLES_PER_CLASS = [50, 100, 150, 200, 250]
datasets = ['bace', 'bbbp'] 
out_file = 'eval_result_pseudo_label.csv'
N_TRIALS = 20

for dataset in datasets:
    for SAMPLE in SAMPLES_PER_CLASS:
        for i in tqdm(range(N_TRIALS)):
            !python pseudo_label/main.py --dataset-name={dataset} --epochs=60 \
            --batch-size=16 --model-name-or-path=shahrukhx01/muv2x-simcse-smole-bert \
            --samples-per-class={SAMPLE} --eval-after=60 --train-log=0 --train-ssl=1 --out-file={out_file}
            !cat {out_file}
```

## Run Pseudo Label Training with Augmentation

```python
from tqdm import tqdm

SAMPLES_PER_CLASS = [50, 100, 150, 200, 250]
N_AUGMENT = [2, 4, 8, 16, 32]
datasets = ['bace', 'bbbp']
out_file = 'eval_result_pseudo_label_augment.csv'
N_TRIALS = 20
EPOCHS = 20

for dataset in datasets:
    for SAMPLE in SAMPLES_PER_CLASS:
        for n_augment in N_AUGMENT:
            for i in tqdm(range(N_TRIALS)):
                !python pseudo_label/main.py --dataset-name={dataset} --epochs={EPOCHS} \
                --batch-size=16 --model-name-or-path=shahrukhx01/muv2x-simcse-smole-bert \
                --samples-per-class={SAMPLE} --eval-after={EPOCHS} --train-log=0 --train-ssl=1 \
                --out-file={out_file} --n-augment={n_augment}
                !cat {out_file}
```

## Run Co-Training

```python
from tqdm import tqdm

SAMPLES_PER_CLASS = [50, 100, 150, 200, 250]
datasets = ['bace', 'bbbp'] 
posterior_thresholds = [0.8, 0.9]
N_TRIALS = 20
out_file = 'eval_result_co_training.csv'

for posterior_threshold in posterior_thresholds:
    for dataset in datasets:
        for SAMPLE in SAMPLES_PER_CLASS:
            for i in tqdm(range(N_TRIALS)):
                !python co_training/main.py --dataset-name={dataset} --epochs=80 \
                --batch-size=8 --model-name-or-path=shahrukhx01/muv2x-simcse-smole-bert \
                --samples-per-class={SAMPLE} --eval-after=80 --train-log=0 --train-ssl=1 \
                --out-file={out_file} --posterior-threshold={posterior_threshold}
                !cat {out_file}        
```
