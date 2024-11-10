# ASR2D: ASR using discrete tokens with decoder-only architecture

The following results have been tested with these packages:
- torch==2.2.0
- triton==2.2.0
- causal-conv1d==1.2.0.post2

## exp/lm_train_mamba_en_bpe10000/decode_test_asr
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_asr/dev_clean|2703|54402|95.0|4.2|0.8|0.4|5.4|49.8|
|decode_lm_asr/dev_other|2864|50948|92.8|6.3|0.9|0.5|7.7|57.8|
|decode_lm_asr/test_clean|2620|52576|94.8|4.2|0.9|0.4|5.5|49.8|
|decode_lm_asr/test_other|2939|52343|92.2|6.7|1.1|0.6|8.4|61.0|

## exp/lm_train_mamba_serial_en_bpe10000/decode_test_asr
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_asr/dev_clean|2703|54402|96.0|3.6|0.4|0.3|4.3|45.5|
|decode_lm_asr/dev_other|2864|50948|93.6|5.8|0.5|0.5|6.9|55.3|
|decode_lm_asr/test_clean|2620|52576|96.0|3.6|0.3|0.4|4.3|46.0|
|decode_lm_asr/test_other|2939|52343|93.1|6.1|0.7|0.6|7.4|58.5|

## exp/lm_train_mamba_parallel_en_bpe10000/decode_test_asr
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_asr/dev_clean|2703|54402|95.9|3.7|0.5|0.3|4.5|46.2|
|decode_lm_asr/dev_other|2864|50948|93.6|5.9|0.5|0.5|6.9|54.9|
|decode_lm_asr/test_clean|2620|52576|96.0|3.6|0.4|0.3|4.3|45.5|
|decode_lm_asr/test_other|2939|52343|93.0|6.3|0.7|0.6|7.6|59.2|

## exp/lm_train_mamba2_en_bpe10000/decode_test_asr
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_asr/dev_clean|2703|54402|95.4|3.7|0.8|0.4|5.0|47.2|
|decode_lm_asr/dev_other|2864|50948|93.0|6.1|0.9|0.5|7.6|56.6|
|decode_lm_asr/test_clean|2620|52576|95.5|3.8|0.7|0.4|4.9|47.1|
|decode_lm_asr/test_other|2939|52343|92.5|6.3|1.2|0.6|8.1|59.5|

## exp/lm_train_mamba2_serial_en_bpe10000/decode_test_asr
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_asr/dev_clean|2703|54402|96.1|3.5|0.4|0.3|4.2|45.8|
|decode_lm_asr/dev_other|2864|50948|93.8|5.7|0.5|0.5|6.7|54.7|
|decode_lm_asr/test_clean|2620|52576|96.2|3.5|0.4|0.4|4.2|45.3|
|decode_lm_asr/test_other|2939|52343|93.3|6.0|0.8|0.6|7.3|57.7|

## exp/lm_train_mamba2_parallel_en_bpe10000/decode_test_asr
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_asr/dev_clean|2703|54402|96.2|3.5|0.3|0.4|4.2|46.1|
|decode_lm_asr/dev_other|2864|50948|93.7|5.8|0.5|0.5|6.8|55.0|
|decode_lm_asr/test_clean|2620|52576|96.2|3.5|0.3|0.4|4.2|44.4|
|decode_lm_asr/test_other|2939|52343|93.2|6.2|0.6|0.6|7.4|59.0|