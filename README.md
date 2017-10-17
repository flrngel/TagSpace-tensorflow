# TagSpace-tensorflow

![model image of TagSpace](https://raw.githubusercontent.com/flrngel/TagSpace-tensorflow/master/resources/tagspace-model.png)

Tensorflow implementation of Facebook **#TagSpace**

You can read more about #TagSpace from [here](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/)

Special thanks to Facebook research team's [Starspace](https://github.com/facebookresearch/Starspace) project, it was really good reference.

## Key Concept

Beside choosing 1000 random negative tag (for performance reason I guess), I choosed worst positive tag, best negative tag for calculating WARP loss. It's not good for performance but since we don't have much tags(labels) as Facebook, it seems okay.

## Usage

Download [ag news dataset](https://github.com/mhjabreel/CharCNN/tree/master/data/ag_news_csv) as below

```
$ tree ./data
./data
└── ag_news_csv
    ├── classes.txt
    ├── readme.txt
    ├── test.csv
    ├── train.csv
    └── train_mini.csv
```

and then

```
$ python train.py
```

## Result

Accuracy 0.89 (ag test data, compare 0.91 from StarSpace with same condition [5 epoch, 10 dim])

## To-do list

- support multiple dataset
- improve performance
- adopt WARP sampling (now is just a WARP loss)
- add Tensorboard metrics
