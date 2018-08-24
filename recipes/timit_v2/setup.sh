
#######################################################################
# Site specific configuration. Override these settings to run on
# you system.

hostname=$(hostname -f)
if [[ "$hostname" == *".fit.vutbr.cz" ]]; then
    timit=/mnt/matylda2/data/TIMIT/timit
    server=matylda5
    parallel_env=sge
    parallel_opts="-l mem_free=200M,ram_free=200M,$server=1"
    parallel_opts_gpu="-l gpu=1,mem_free=1G,ram_free=1G"
elif [[ "$hostname" = *"clsp.jhu.edu" ]]; then
    timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT
    parallel_env=sge
    parallel_opts="-l mem_free=200M,ram_free=200M,hostname=b*|c*"
    parallel_opts_gpu="-l gpu=1,mem_free=1G,ram_free=1G,hostname=b1[123456789]*|c*"
else
    echo "Unkown location configuration. Please update the"
    echo "\"setup.sh\" file."
    exit 1
fi


#######################################################################
# Directory structure.

confdir=$(pwd)/conf
datadir=$(pwd)/data
langdir=$datadir/lang
expdir=$(pwd)/exp


#######################################################################
# Features extraction.

fea_njobs=10
fea_parallel_opts="$parallel_opts"
fea_conf=$confdir/features.yml


#######################################################################
# HMM-GMM model parameters.

hmm_emission_conf=$confdir/hmm_gmm/hmm.yml
hmm_dir=$expdir/hmm_gmm
hmm_align_njobs=20
hmm_align_parallel_opts="$parallel_opts"
hmm_align_iters="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29"
hmm_train_iters=30
hmm_train_emissions_lrate=0.1
hmm_train_emissions_batch_size=400
hmm_train_emissions_epochs=10
hmm_train_emissions_opts="--fast-eval --use-gpu"
hmm_train_parallel_opts="$parallel_opts_gpu"
hmm_decode_njobs=2
hmm_decode_parallel_opts="$parallel_opts"



#######################################################################
# VAE-HMM model.

vae_hmm_confdir=$confdir/vae_hmm
vae_hmm_dir=$expdir/vae_hmm
vae_hmm_encoder_conf=$vae_hmm_confdir/encoder.yml
vae_hmm_decoder_conf=$vae_hmm_confdir/decoder.yml
vae_hmm_nflow_conf=$vae_hmm_confdir/normalizing_flow.yml
vae_hmm_nnet_width=512
vae_hmm_latent_dim=30
vae_hmm_hmm_conf=$vae_hmm_confdir/hmm.yml
vae_hmm_encoder_cov_type=isotropic
vae_hmm_decoder_cov_type=diagonal
vae_hmm_align_njobs=10
vae_hmm_align_sge_opts=""
vae_hmm_align_epochs="2 3 4 5 6 7 8 9 10"
vae_hmm_train_iters=10
vae_hmm_train_epochs_per_iter=50
vae_hmm_train_warmup_iters=1
vae_hmm_train_emissions_lrate=1e-1
vae_hmm_train_emissions_nnet_lrate=1e-3
vae_hmm_train_emissions_batch_size=400
vae_hmm_train_emissions_opts="--fast-eval --use-gpu"
vae_hmm_train_emissions_sge_opts="-l gpu=1,hostname=*face*"


#######################################################################
# Score options.

remove_sym="" # Support multiple symbol, e.g. "sil spn nsn"
duplicate="no" # Do not allow adjacent duplicated phones. Only effective at scoring stage.
phone_48_to_39_map=$langdir/phones_48_to_39.txt

