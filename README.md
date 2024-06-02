## 0.环境

- tensorflow==1.15.0
  python==3.7
  环境安装：conda env create -f enviroment.yaml
- python==3.8
  tensorboard：tensorboard --logdir=experiments/my_experiment

## 1.trinet_embed

```
python trinet_embed.py expercondiments\trinet_embed_experiments\files.txt experiments\trinet_embed_experiments\trinet-market1501.npz > experiments\trinet_embed_experiments\embeddings.csv
```

这个我的环境运行不了一直报错

## 2.train

**Linux**：market1501_train.sh market1501\Market-1501-v15.09.15 resnet_v2_50_2017_04_14 experiments/my_experiment

**window**：

- 从头训练：

```
rm .\experiments\my_experiment\*
python train.py --train_set data/market1501_train.csv --image_root market1501\Market-1501-v15.09.15 --experiment_root experiments\my_experiment --initial_checkpoint resnet_v1_50_2016_08_28\resnet_v1_50.ckpt --batch_p 8 --batch_k 4 --loading_threads 1 --checkpoint_frequency 100 --detailed_logs
```

- 继续训练

```
python train.py --experiment_root ./experiments/my_experiment --resume
```

## 3.evaluation

**embed**

```
python embed.py --experiment_root ./experiments/my_experiment --dataset data/market1501_query.csv --filename market1501_query_embeddings_augmented.h5 --flip_augment --crop_augment five --aggregator mean
python embed.py --experiment_root ./experiments/my_experiment --dataset data/market1501_test.csv --filename market1501_test_embeddings.h5
```

**evaluate**

```
python ./evaluate.py --excluder market1501 --query_dataset data/market1501_query.csv --query_embeddings ./experiments/my_experiment/market1501_query_embeddings_augmented.h5 --gallery_dataset data/market1501_test.csv --gallery_embeddings ./experiments/my_experiment/market1501_test_embeddings.h5 --metric euclidean --filename ./experiments/my_experiment/market1501_evaluation.json
```
