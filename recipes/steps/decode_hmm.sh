#!/bin/bash

if [ $# != 1 ]; then
    echo "$0: decode_setup.sh"
    exit 1
fi

setup=$1
stage=-1
. $setup

if [ ! -d $decode_dir ]; then
    mkdir -p $decode_dir || exit 1
fi
cp $setup $decode_dir

if [ $stage -le 0 ]; then
    echo "Creating state labels for test data"
    python3 steps/prepare_labels.py $phonelist $trans $nstate_per_phone
    zip -j $decode_data_dir/states.int.npz $decode_data_dir/tmp/*.npy
    rm -r $decode_data_dir/tmp
fi


python3 steps/decode_hmm.py $model $decode_dir $feats \
                            $trans $phonelist $nstate_per_phone \
                            --gamma $gamma \
                            --phone_39 $phone_39 \
                            --remove_sys $remove_sys \
                            --score \
                            > $decode_dir/decode.log 2>&1
#                            --use-gpu

