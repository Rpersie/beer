#!/bin/sh

gpu=  # Empty variable means we don't use the GPU.
lograte=100
epochs=10
kl_weight=1.
lrate=.1
lrate_nnet=1e-3
train_cmd=utils/train-vae-discrete-latent-model.py

usage() {
echo "Usage: $0 [options] <sge-options> <init-model> <dbstats> <archives> <outdir>"
}

help() {
echo "\
Train a Variational Auto-Encoder model with discrete latent
variable prior (i.e. GMM or similar).

"
usage
echo "
Options:
  -h --help        show this message
  --use-gpu        use gpu for the training
  --unsupervised   unsupervised training (ignore the labels if
                   any)
  --lograte        log message rate
  --epochs         number of epochs for the training
  --kl-weight      weight of the KL divergence of the ELBO (1.)
  --lrate          learning rate for the latent model
  --lrate-nnet     learning for the encoder/decoder networks

Example:
  \$ $0 \\
            --lograte=100 \\
            --epochs=10 \\
            --lrate=.1 \\
            --lrate-nnet=1e-3 \\
            -- \\
            \"-l mem_free=1G,ram_free=1G\" \\
             /path/to/init.mdl \\
             /path/to/dbstats.npz \\
             /path/to/archives/ expdir

Note the double hyphens \"--\" to avoid problem when parsing
the SGE option \"-l ...\".

The final model is written in \"<outdir>/final.mdl\".
"
}

# Parsing optional arguments.
while [ $# -ge 0 ]; do
    param=$(echo $1 | awk -F= '{print $1}')
    optname=$(echo ${param} | sed 's/--//g' | sed 's/-/_/g')
    value=`echo $1 | awk -F= '{print $2}'`
    case $param in
        -h | --help)
            help
            exit
            ;;
        --use-gpu)
            gpu="--use-gpu"
            shift
            ;;
        --unsupervised)
            train_cmd=utils/train-vae-model.py
            shift
            ;;
        --lograte | \
        --epochs | \
        --kl-weight | \
        --lrate | \
        --lrate-nnet)
            eval ${optname}=${value}
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            usage
            exit 1
            ;;
        *)
            break
    esac
done

# Parsing mandatory arguments.
if [ $# -ne 5 ]; then
    usage
    exit 1
fi

sge_options=$1
init_model=$2
dbstats=$3
archives=$4
root=$5

# Build the output directory followin the parameters.
outdir="${root}/epochs${epochs}_lrate${lrate}_lratennet${lrate_nnet}"
mkdir -p ${outdir}


training_options="\
--epochs ${epochs}  \
${gpu}  \
--lrate ${lrate}  \
--lrate-nnet ${lrate_nnet} \
--logging-rate ${lograte}  \
--dir-tmp-models ${outdir}/training \
--kl-weight ${kl_weight} \
"

if [ ! -f "${outdir}/.done" ]; then
    echo "Training..."
    # Command to submit to the SGE.
    cmd="python "${train_cmd}" \
        ${training_options} \
        ${dbstats} \
        ${archives} \
        ${init_model} \
        ${outdir}/final.mdl"

    # Clear the log file.
    rm -f ${outdir}/training/sge.log

    # Submit the command to the SGE.
    qsub \
        ${sge_options}\
        -wd $(pwd)\
        -j y \
        -sync y \
        -o ${outdir}/training/sge.log \
        utils/job.qsub \
        "${cmd}" || exit 1

    cp "${outdir}/final.mdl" "${root}"

    date > "${outdir}/.done"
else
    echo "Model already trained. Skipping."
fi

