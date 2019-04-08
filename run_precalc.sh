#!/bin/bash

usage() {
    echo "usage: $0 <data_dir>"
}

if [ "$#" -ne 1 ]; then
    usage
    exit
fi

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
DOWNLOAD_SCRIPT=$SCRIPT_DIR/download_external_data.sh
EMBEDDING_SCRIPT=$SCRIPT_DIR/precalc_embeddings.py
BERT_SCRIPT=$SCRIPT_DIR/generate_bert_features.py
BASELINE_SCRIPT=$SCRIPT_DIR/preprocess_baseline_features.py

DATA_DIR=$1

EXTERNAL_DATA_DIR=$DATA_DIR/external

echo Downloading external data...
$DOWNLOAD_SCRIPT $EXTERNAL_DATA_DIR

DATASET_DIR=$EXTERNAL_DATA_DIR/dataset
TRAIN_SUFFIX=train
TRAIN_DIR=$DATASET_DIR/$TRAIN_SUFFIX
DEV_SUFFIX=dev
DEV_DIR=$DATASET_DIR/$DEV_SUFFIX
FASTTEXT_EN_MODEL=$EXTERNAL_DATA_DIR/embeddings/en/wiki.en.bin
FASTTEXT_DE_MODEL=$EXTERNAL_DATA_DIR/embeddings/de/wiki.de.bin
BASELINE_DIR=$EXTERNAL_DATA_DIR/baseline/en_de/word_features

PRECALC_DIR=$DATA_DIR/precalc

EMBEDDINGS_DIR=$PRECALC_DIR/embeddings
EMBEDDINGS_EN_DEST=$EMBEDDINGS_DIR/en.pkl
EMBEDDINGS_DE_DEST=$EMBEDDINGS_DIR/de.pkl
ENGLISH_TEXTS="$TRAIN_DIR/dev.src $DEV_DIR/train.src"
GERMAN_TEXTS="$TRAIN_DIR/dev.mt $DEV_DIR/train.mt"

mkdir -p $EMBEDDINGS_DIR

if [ ! -f "$EMBEDDINGS_EN_DEST" ]; then
    echo Precalcing English embeddings...
    $EMBEDDING_SCRIPT --fasttext-model=$FASTTEXT_EN_MODEL --output-file=$EMBEDDINGS_EN_DEST $ENGLISH_TEXTS
else
    echo Skipping English embeddings precalc...
fi

if [ ! -f "$EMBEDDINGS_DE_DEST" ]; then
    echo Precalcing German embeddings...
    $EMBEDDING_SCRIPT --fasttext-model=$FASTTEXT_DE_MODEL --output-file=$EMBEDDINGS_DE_DEST $GERMAN_TEXTS
else
    echo Skipping German embeddings precalc...
fi

BASELINE_DEST=$PRECALC_DIR/baseline
BASELINE_VOCABS_DEST=$BASELINE_DEST/vocabs.pkl
TRAIN_BASELINE_DEST=$BASELINE_DEST/train.pkl
DEV_BASELINE_DEST=$BASELINE_DEST/dev.pkl
TRAIN_BASELINE_SRC=$BASELINE_DIR/train.nmt.features
DEV_BASELINE_SRC=$BASELINE_DIR/dev.nmt.features

mkdir -p $BASELINE_DEST

if [[ ! -f "$TRAIN_BASELINE_DEST" || ! -f "$BASELINE_VOCABS_DEST" ]]; then
    if [ -f "$DEV_BASELINE_DEST" ]; then
        rm -rf $DEV_BASELINE_DEST
    fi
    echo Preprocessing baseline features for train...
    $BASELINE_SCRIPT --features-file=$TRAIN_BASELINE_SRC --features-out=$TRAIN_BASELINE_DEST --vocabs-out=$BASELINE_VOCABS_DEST
else
    echo Skipping baseline preprocessing for train...
fi

if [ ! -f "$DEV_BASELINE_DEST" ]; then
    echo Preprocessing baseline features for dev...
    $BASELINE_SCRIPT --features-file=$DEV_BASELINE_SRC --features-out=$DEV_BASELINE_DEST --vocabs-file=$BASELINE_VOCABS_DEST
else
    echo Skipping baseline preprocessing for dev...
fi

BERT_DEST=$PRECALC_DIR/bert
TRAIN_BERT_DEST=$BERT_DEST/train.pkl
DEV_BERT_DEST=$BERT_DEST/dev.pkl

mkdir -p $BERT_DEST

if [ ! -f "$TRAIN_BERT_DEST" ]; then
    echo Generating bert features for train...
    $BERT_SCRIPT --dataset-path=$TRAIN_DIR --dataset-name=train --out-file=$TRAIN_BERT_DEST
else
    echo Skipping bert generation for train...
fi

if [ ! -f "$DEV_BERT_DEST" ]; then
    echo Generating bert features for dev...
    $BERT_SCRIPT --dataset-path=$DEV_DIR --dataset-name=dev --out-file=$DEV_BERT_DEST
else
    echo Skipping bert generation for dev...
fi

DATASET_DEST=$DATA_DIR/dataset
TRAIN_DEST=$DATASET_DEST/$TRAIN_SUFFIX
DEV_DEST=$DATASET_DEST/$DEV_SUFFIX

if [ -d $DATASET_DEST ]; then
    rm -rf $DATASET_DEST
fi
echo Creating extended dataset...
cp -r $DATASET_DIR $DATASET_DEST
ln -s $TRAIN_BASELINE_DEST $TRAIN_DEST/train.baseline
ln -s $TRAIN_BERT_DEST $TRAIN_DEST/train.bert
ln -s $DEV_BASELINE_DEST $DEV_DEST/dev.baseline
ln -s $DEV_BERT_DEST $DEV_DEST/dev.bert
