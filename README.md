# TagSpace-tensorflow

Tensorflow implementation of Facebook **#TagSpace**

You can read more about #TagSpace from (here)[https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/]

Special thanks to Facebook research team's (Starspace)[https://github.com/facebookresearch/Starspace] project, it was really good reference.

## Key Concept

Beside choosing 1000 random negative tag (for performance reason I guess), I choosed worst positive tag, best negative tag.

## Usage

Download (ag news dataset)[https://github.com/mhjabreel/CharCNN/tree/master/data/ag_news_csv] as below

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
$ python model.py
```

## To-do list

- Clean up messy code
- improve Tokenizer
- support Stackoverflow dataset
- improve performance
- add Tensorboard metrics
- add Korean
