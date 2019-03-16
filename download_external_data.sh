#!/bin/bash

usage() {
    echo "usage: $0 <download_dir>"
}

if [ "$#" -ne 1 ]; then
    usage
    exit
fi

DOWNLOAD_DIR=$1
mkdir -p $DOWNLOAD_DIR

TEMP_DIR=$DOWNLOAD_DIR/temp
mkdir -p $TEMP_DIR

DATASET_URL=https://deep-spin.github.io/docs/data/wmt2019_qe/task1_en-de_traindev.tar.gz
DATASET_TEMP=$TEMP_DIR/dataset.tar.gz
DATASET_DEST=$DOWNLOAD_DIR/dataset

if [ ! -d "$DATASET_DEST" ]; then
    mkdir -p $DATASET_DEST
    echo Downloading dataset...
    curl $DATASET_URL -o $DATASET_TEMP
    mkdir -p $DATASET_DEST
    echo Extracting...
    tar xzf $DATASET_TEMP -C $DATASET_DEST
else
    echo Skipping dataset...
fi

EMBEDDINGS_EN_URL=https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
EMBEDDINGS_EN_TEMP=$TEMP_DIR/ft_en.zip
EMBEDDINGS_EN_DEST=$DOWNLOAD_DIR/embeddings/en

if [ ! -d "$EMBEDDINGS_EN_DEST" ]; then
    echo Downloading English embeddings...
    curl $EMBEDDINGS_EN_URL -o $EMBEDDINGS_EN_TEMP
    mkdir -p $EMBEDDINGS_EN_DEST
    echo Extracting...
    unzip -qq $EMBEDDINGS_EN_TEMP -d $EMBEDDINGS_EN_DEST
else
    echo Skipping English embeddings...
fi

EMBEDDINGS_DE_URL=https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.zip
EMBEDDINGS_DE_TEMP=$TEMP_DIR/ft_de.zip
EMBEDDINGS_DE_DEST=$DOWNLOAD_DIR/embeddings/de

if [ ! -d "$EMBEDDINGS_DE_DEST" ]; then
    echo Downloading German embeddings...
    curl $EMBEDDINGS_DE_URL -o $EMBEDDINGS_DE_TEMP
    mkdir -p $EMBEDDINGS_DE_DEST
    echo Extracting...
    unzip -qq $EMBEDDINGS_DE_TEMP -d $EMBEDDINGS_DE_DEST
else
    echo Skipping German embeddings...
fi

echo Cleaning up...
rm -rf $TEMP_DIR

echo Done.