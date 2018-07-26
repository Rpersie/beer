#!/bin/bash

if [ $# != 3 ];then
    echo "Usage: $0 srcdir num_of_utt tgtdir"
    exit 1
fi

srcdir=$1
num_utts=$2
tgtdir=$3

if [ ! -d $tgtdir ]; then
    mkdir -p $tgtdir || exit "Cannot mkdir $tgtdir";
fi

mkdir -p $tgtdir/tmp_ft || exit 1;
mkdir -p $tgtdir/tmp_lab || exit 1;

python3 utils/random_choice_subset.py $srcdir $num_utts $tgtdir

zip -j $tgtdir/feats_transformed.npz $tgtdir/tmp_ft/*.npy
zip -j $tgtdir/states.int.npz $tgtdir/tmp_lab/*.npy

rm -r $tgtdir/tmp_*
