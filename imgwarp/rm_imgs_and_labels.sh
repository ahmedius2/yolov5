#!/bin/sh

DATASET_DIR=$1
for split in train valid val test
do
	rm -f $DATASET_DIR/$split/images/*
	rm -f $DATASET_DIR/$split/labels/*
done
