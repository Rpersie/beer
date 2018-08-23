#!/bin/bash

if [ $# -ne 2 ];then
    echo "Read Kaldi feats and save to npz files."
    echo "$0: <scpfile> <tgtdir>"
    exit 1
fi

scpfile=$1
tgtdir=$2

if [ ! -d $tgtdir ];then
    mkdir -p $tgtdir/tmp
else
    echo "$tgtdir already exists."
    exit 1
fi

python utils/read_kaldi_feats.py $scpfile $tgtdir/tmp 2>&1 | grep -v LOG
zip -j $tgtdir/feats.npz $tgtdir/tmp/*.npy > /dev/null 
rm -r $tgtdir/tmp

    
