# ASR2D: ASR using discrete tokens with decoder-only architecture

The following results have been tested with these packages:
- torch==2.2.0
- triton==2.2.0
- causal-conv1d==1.2.0.post2

## exp/lm_train_mamba2_parallel_en_bpe10000/decode_test_asr
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_asr/dev_clean|2703|54402|98.0|1.8|0.2|0.2|2.2|27.8|
|decode_lm_asr/dev_other|2864|50948|95.8|3.8|0.4|0.4|4.6|41.6|
|decode_lm_asr/test_clean|2620|52576|97.9|1.9|0.3|0.3|2.4|28.8|
|decode_lm_asr/test_other|2939|52343|95.8|3.7|0.5|0.4|4.6|42.6|