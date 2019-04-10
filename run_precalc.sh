#!/bin/bash

usage() {
    echo "usage: $0 <data_dir> [--bert-tokens]"
}

if [ "$#" -lt 1 ]; then
    usage
    exit
fi

if [[ "$2" != "" && "$2" != "--bert-tokens" ]]; then
    usage
    exit
fi

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
DOWNLOAD_SCRIPT=$SCRIPT_DIR/download_external_data.sh
EMBEDDING_SCRIPT=$SCRIPT_DIR/precalc_embeddings.py
BASELINE_SCRIPT=$SCRIPT_DIR/preprocess_baseline_features.py
BERT_SCRIPT=$SCRIPT_DIR/generate_bert_features.py

# args: data-dir
download_external_data() {
    $DOWNLOAD_SCRIPT $1
}

# args: fasttext-model, output-file, bert-tokens, text-files
precalc_embeddings() {
    $EMBEDDING_SCRIPT --fasttext-model=$1 --output-file=$2 $3 $4
}

# args: features-file, features-out, vocabs-option
preprocess_baseline_features() {
    $BASELINE_SCRIPT --features-file=$1 --features-out=$2 $3
}

# args: dataset-path, dataset-name, out-file, bert-tokens
generate_bert_features() {
    $BERT_SCRIPT --dataset-path=$1 --dataset-name=$2 --out-file=$3 $4
}

DATA_DIR="$(readlink -f "$1")"
BERT_TOKENS=$2

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
ENGLISH_TEXTS="$TRAIN_DIR/train.src $DEV_DIR/dev.src"
GERMAN_TEXTS="$TRAIN_DIR/train.mt $DEV_DIR/dev.mt"

mkdir -p $EMBEDDINGS_DIR

if [ ! -f "$EMBEDDINGS_EN_DEST" ]; then
    echo Precalcing English embeddings...
    precalc_embeddings $FASTTEXT_EN_MODEL $EMBEDDINGS_EN_DEST "$BERT_TOKENS" "$ENGLISH_TEXTS"
else
    echo Skipping English embeddings precalc...
fi

if [ ! -f "$EMBEDDINGS_DE_DEST" ]; then
    echo Precalcing German embeddings...
    precalc_embeddings $FASTTEXT_DE_MODEL $EMBEDDINGS_DE_DEST "$BERT_TOKENS" "$GERMAN_TEXTS"
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
    preprocess_baseline_features $TRAIN_BASELINE_SRC $TRAIN_BASELINE_DEST --vocabs-out=$BASELINE_VOCABS_DEST
else
    echo Skipping baseline preprocessing for train...
fi

if [ ! -f "$DEV_BASELINE_DEST" ]; then
    echo Preprocessing baseline features for dev...
    preprocess_baseline_features $DEV_BASELINE_SRC $DEV_BASELINE_DEST --vocabs-file=$BASELINE_VOCABS_DEST
else
    echo Skipping baseline preprocessing for dev...
fi

BERT_DEST=$PRECALC_DIR/bert
TRAIN_BERT_DEST=$BERT_DEST/train.pkl
DEV_BERT_DEST=$BERT_DEST/dev.pkl

mkdir -p $BERT_DEST

if [ ! -f "$TRAIN_BERT_DEST" ]; then
    echo Generating bert features for train...
    generate_bert_features $TRAIN_DIR train $TRAIN_BERT_DEST "$BERT_TOKENS"
else
    echo Skipping bert generation for train...
fi

if [ ! -f "$DEV_BERT_DEST" ]; then
    echo Generating bert features for dev...
    generate_bert_features $DEV_DIR dev $DEV_BERT_DEST "$BERT_TOKENS"
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
