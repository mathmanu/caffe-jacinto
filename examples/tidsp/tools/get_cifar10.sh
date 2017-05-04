#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

rm -rf ./data/cifar10
mkdir ./data/cifar10
echo "Downloading..."

wget --no-check-certificate http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

echo "Unzipping..."

tar -xf cifar-10-binary.tar.gz && rm -f cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* ./data/cifar10 && rm -rf cifar-10-batches-bin

echo "Done."
