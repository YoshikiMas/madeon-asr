#!/usr/bin/env bash

# This part should follow the ASR2 recipe
train_set="train_960"
dev_set="dev"
test_sets="test_clean test_other dev_clean dev_other"
speed_perturb_factors="0.9 1.0 1.1"
asr2_data_feats="../asr2/dump/extracted/wavlm_large/layer21"
src_case="rm"
tgt_case="ts"
src_lang="wavlm_large_21_km2000"
tgt_lang="en"


# This part is for ASR2D
# The original training set is used for BPE even if speed perturbation is perfromed.
data_feats="dump/raw"
token_type=bpe
bpe_train_text="data/${train_set}/bpe_text"
text_upsampling_factor=4


dsets="${train_set} ${dev_set} ${test_sets}"
if [ -n "${speed_perturb_factors}" ]; then
   dsets+=" ${train_set}_sp"
fi

for dset in $dsets; do
   python3 local/prepare_lm_data_from_asr2.py \
      --data_feats $data_feats \
      --asr2_data_feats $asr2_data_feats \
      --dset $dset \
      --src_case $src_case \
      --tgt_case $tgt_case \
      --src_lang $src_lang \
      --tgt_lang $tgt_lang
done

for dset in $test_sets; do
   python3 local/prepare_lm_test_from_asr2.py \
      --data_feats $data_feats \
      --dset $dset
done

if [ "${token_type}" = bpe ]; then
   python3 local/prepare_bpe_text.py \
      -i "${data_feats}/${train_set}/lm_text" \
      -o ${bpe_train_text} \
      --upsampling_factor $text_upsampling_factor
fi

if [ -n "${speed_perturb_factors}" ]; then
   cat  "${data_feats}/${train_set}_sp/lm_text" | awk ' { if( NF != 1 ) print $0; } ' > "dump/raw/lm_train.txt"
else
   cat  "${data_feats}/${train_set}/lm_text" | awk ' { if( NF != 1 ) print $0; } ' > "dump/raw/lm_train.txt"
fi