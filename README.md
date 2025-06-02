## Requirements
- Python version 3.11

## Installation
Install the dependencies:
``` sh
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Start training
To start training, use the following command:

1. Movielens-100K
``` sh
python train.py --dataset 100k --lr 0.1 --l2_regularization 1e-6 --num_round 300
```

2. Movielens-1M
``` sh
python train.py --dataset ml-1m --lr 0.1 --l2_regularization 1e-6 --num_round 300
```

3. Lastfm-2K
``` sh
python train.py --dataset lastfm-2k --lr 0.1 --l2_regularization 1e-6 --num_round 300
```

4. HetRec2011
``` sh
python train.py --dataset hetrec --lr 0.1 --l2_regularization 1e-6 --num_round 300
```