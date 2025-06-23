#!/bin/sh

DIR=$1
FOLDS=5
BASE_PREFIX=$2

BASE="${DIR}/${BASE_PREFIX}base.yaml"
echo "From $BASE"
for i in $(seq 0 $(($FOLDS-1)))
do
	BASE="${DIR}/${BASE_PREFIX}base.yaml"
    echo "Generating ${DIR}/${BASE_PREFIX}$i.yaml ..."
	sed "s/kfold_idx: 0/kfold_idx: "$i"/g" $BASE | \
        sed "s|0/metrics|"$i"/metrics|g" | \
        sed "s|0/checkpoints|"$i"/checkpoints|g" | \
        sed -E "s|^([ \t]+save_dir: .*/)[^/]+/?$|\1"$i"/|" | \
        sed -E "s|^([ \t]+name: .*/)[^/]+/?$|\1"$i"/|" > "${DIR}/"${BASE_PREFIX}${i}".yaml"	

done
