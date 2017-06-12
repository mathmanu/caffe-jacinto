#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

OUT=./data
DATA=./data/cifar10
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $OUT/cifar10_train_$DBTYPE $OUT/cifar10_test_$DBTYPE

../../build/examples/cifar10/convert_cifar_data.bin $DATA $OUT $DBTYPE

echo "Done."
