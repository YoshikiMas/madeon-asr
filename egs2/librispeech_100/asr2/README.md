# ASR2 recipe for LibriSpeech100
## Related work:
   Chang, Xuankai, et al. "Exploration of Efficient End-to-End ASR using Discretized Input from Self-Supervised Learning." InterSpeech 2023.
   <details>
   <summary>bib info</summary>

   ```
   @article{chang2023exploration,
        title={Exploration of Efficient End-to-End ASR using Discretized Input from Self-Supervised Learning},
        author={Chang, Xuankai and Yan, Brian and Fujita, Yuya and Maekaku, Takashi and Watanabe, Shinji},
        journal={arXiv preprint arXiv:2305.18108},
        year={2023}
   }
   ```
   </details>

# RESULTS
## Environments
- date: `Tue May 21 05:43:16 UTC 2024`
- python version: `3.10.6 (main, Mar 10 2023, 10:55:28) [GCC 11.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0+cu121`
- Git hash: `9e4a1c38faf8a22fc5a6dbcbf50d7b5354d6c653`
  - Commit date: `Thu May 16 10:52:28 2024 +0000`

## E-branchformer + Mamba w/o Subword regularization
- ASR config: [conf/tuning/train_discrete_asr_e_branchformer_mamba.yaml](conf/tuning/train_discrete_asr_e_branchformer_mamba.yaml)
- kmeans_feature: `wavlm_large/21`
- nclusters: 2000
- src_nbpe: 6000
- tgt_nbpe: 5000
- src_case: "rm"
- tgt_case: "ts"

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|54402|96.6|3.2|0.2|0.4|3.8|43.7|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|50948|94.2|5.5|0.4|0.5|6.4|53.0|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|52576|96.6|3.2|0.2|0.4|3.8|42.7|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|52343|93.9|5.7|0.5|0.6|6.8|57.1|
|decode_wo_ctc_05_asr_model_valid.acc.ave/dev_clean|2703|54402|96.2|3.5|0.4|1.6|5.5|43.9|
|decode_wo_ctc_05_asr_model_valid.acc.ave/dev_other|2864|50948|93.7|5.4|0.8|0.8|7.1|53.4|
|decode_wo_ctc_05_asr_model_valid.acc.ave/test_clean|2620|52576|96.0|3.3|0.7|1.1|5.2|42.7|
|decode_wo_ctc_05_asr_model_valid.acc.ave/test_other|2939|52343|93.4|5.8|0.8|1.3|8.0|56.9|

## E-branchformer + Transformer w/o Subword regularization
- ASR config: [conf/tuning/train_discrete_asr_e_branchformer_transformer.yaml](conf/tuning/train_discrete_asr_e_branchformer_transformer.yaml)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|54402|96.4|3.4|0.2|0.4|4.0|44.6|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|50948|94.0|5.6|0.4|0.6|6.5|54.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|52576|96.5|3.2|0.3|0.4|3.9|42.5|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|52343|93.8|5.8|0.5|0.6|6.8|56.4|
|decode_wo_ctc_05_asr_model_valid.acc.ave/dev_clean|2703|54402|95.9|3.5|0.7|0.5|4.6|44.5|
|decode_wo_ctc_05_asr_model_valid.acc.ave/dev_other|2864|50948|93.5|5.6|0.9|0.6|7.2|54.2|
|decode_wo_ctc_05_asr_model_valid.acc.ave/test_clean|2620|52576|95.6|3.3|1.0|0.6|4.9|43.1|
|decode_wo_ctc_05_asr_model_valid.acc.ave/test_other|2939|52343|93.2|5.7|1.1|0.8|7.6|57.0|

# E-Branchformer ASR2 Discrete tokens with WavLM_large_Layer21_Kmeans2000_nBPE5000 (7 hours for 70epochs with A6000 x 1)

## Environments
- date: `Sun Jul 23 11:19:30 CEST 2023`
- python version: `3.10.8 (main, Nov 14 2022, 00:00:00) [GCC 11.3.1 20220421 (Red Hat 11.3.1-3)]`
- espnet version: `espnet 202304`
- pytorch version: `pytorch 1.13.1+cu117`
- Git hash: `64a1cabc6e7fe4fd22d46b788cec29ba6a37801e`
  - Commit date: `Sat Jul 22 23:17:44 2023 +0200`
  - ASR config: [conf/train_discrete_asr_e_branchformer1_1gpu.yaml](conf/train_discrete_asr_e_branchformer1_1gpu.yaml)
  - Decode config: [conf/decode_ctc0.3.yaml](conf/decode_ctc0.3.yaml)
  - Pretrained model: [https://huggingface.co/espnet/akreal_ls100_asr2_e_branchformer1_1gpu_raw_wavlm_large_21_km2k_bpe_rm6k_bpe_ts5k_sp](https://huggingface.co/espnet/akreal_ls100_asr2_e_branchformer1_1gpu_raw_wavlm_large_21_km2k_bpe_rm6k_bpe_ts5k_sp)

## exp/asr_train_discrete_asr_e_branchformer1_1gpu_raw_wavlm_large_21_km2000_bpe_rm6000_bpe_ts5000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|54402|96.5|3.3|0.2|0.4|3.8|42.5|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|50948|93.9|5.7|0.4|0.5|6.7|54.4|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|52576|96.5|3.3|0.2|0.4|3.9|43.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|52343|93.6|5.9|0.5|0.6|7.0|56.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|288456|99.0|0.6|0.4|0.4|1.4|42.5|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|265951|97.8|1.3|0.9|0.7|2.9|54.4|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|281530|99.0|0.6|0.5|0.4|1.4|43.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|272758|97.9|1.2|0.9|0.8|2.8|56.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|69558|94.8|3.2|2.0|0.5|5.6|42.5|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|64524|91.3|5.7|3.0|1.2|9.9|54.4|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|66983|94.8|3.2|2.0|0.5|5.7|43.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|66650|91.4|5.7|2.9|1.2|9.7|56.2|
