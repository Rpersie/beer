#!/bin/bash

if [ $# -ne 3 ];then
    echo "Prepare state id transcriptions and compute data statistics"
    echo "$0: phonemes_set, data_dir num_states_per_phone"
    echo "eg: $0 data/lang/phones.txt data/train 3"
    exit 1
fi


phones=$1
datadir=$2
nstate_per_phone=$3
rootdir=`pwd`


if [ -d $datadir/tmp ]; then
    rm -r $datadir/tmp
fi

python3 steps/prepare_labels.py $langdir/phones.txt \
    $datadir/phones.text \
    $nstate_per_phone
python3 steps/accumulate_data_stats.py $datadir/feats_transformed.npz $datadir/feats_stats.npz
zip -j $datadir/states.int.npz $datadir/tmp/*.npy > /dev/null 2>&1
rm -r $datadir/tmp
