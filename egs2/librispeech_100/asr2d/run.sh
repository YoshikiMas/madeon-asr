#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"
speed_perturb_factors="0.9 1.0 1.1"

nbpe=10000
km_dir=""
lm_config=conf/train_mamba2_parallel.yaml
lm_inference_asr_config=conf/decode_lm_asr.yaml
lm_inference_tts_config=""

bpe_train_text="data/${train_set}/bpe_text"

# prepare discrete tokens for LM training
if [ ! -d dump ]; then
    # Before proceeding, please check the run.sh options on the ASR2 side.
    # In particular, the following parameters are important for optimal performance as with asr2:
    # - kmeans_feature
    # - nclusters
    asr2_opts="--stop-stage 6"
    (cd ../asr2 && ./run.sh ${asr2_opts})
    ./local/asr2_to_asr2d.sh
fi

./lm.sh \
    --kmeans_feature "wavlm_large/21" \
    --nclusters 2000 \
    --learn_kmeans false \
    --stage 1 \
    --stop_stage 9 \
    --skip_stages "1 2 3 4 8 10 11 12 13 " \
    --num_splits_lm 1 \
    --nj 16 \
    --ngpu 1 \
    --gpu_inference true \
    --inference_nj 16 \
    --lang en \
    --token_type bpe \
    --nbpe "${nbpe}" \
    --bpe_nlsyms data/nlsyms.txt \
    --bpe_train_text "${bpe_train_text}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_lm "valid.acc.ave.pth" \
    --km_dir "${km_dir}" \
    --lm_inference_asr_config "${lm_inference_asr_config}" \
    --lm_inference_tts_config "${lm_inference_tts_config}" \
    --lm_test_text_asr "" \
    --lm_test_text_tts "" \
    --lm_test_text_textlm "" \
    --lm_test_text_speechlm "" "$@"
