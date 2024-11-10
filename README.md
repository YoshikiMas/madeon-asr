# Mamba-based Decoder-Only Approach with Bidirectional Speech Modeling for Speech Recognition

This repository is for a Mamba-based decoder-only approach (MADEON) for speech recognition proposed in the following SLT 2024 paper:
```
@InProceedings{Masuyama2024SLT,
  author    =  {Masuyama, Yoshiki and Miyazaki, Koichi and Murata, Masato},
  title     =  {Mamba-based Decoder-Only Approach with Bidirectional Speech Modeling for Speech Recognition},
  booktitle =  {Proc. IEEE Spoken Language Technology Workshop (SLT)},
  year      =  2024,
  month     =  dec
}
```

```
Abstract:
Selective state space models (SSMs) represented by Mamba have demonstrated their computational efficiency and promising outcomes in various tasks, including automatic speech recognition (ASR). Mamba has been applied to ASR task with the attention-based encoder-decoder framework, where the cross-attention mechanism between encoder and decoder remains. This paper explores the capability of Mamba as the decoder-only architecture in ASR task. Our MAmba-based DEcoder-ONly approach (MADEON) consists of a single decoder that takes speech tokens as a condition and predicts text tokens in an autoregressive manner. To enhance MADEON, we further propose speech prefixing that performs bidirectional processing on speech tokens, which enriches the contextual information in the hidden states. Our experiments show that MADEON significantly outperforms a non-selective SSM. The combination of speech prefixing and the recently proposed Mamba-2 yields comparable performance to Transformer-based models on large datasets.
```

We currently support a comparison between variants of MADEON on [LibriSpeech 100h](egs2/librispeech_100/asr2d/).

## Environment setup
Our implementation is built on top of [ESPnet](https://espnet.github.io/espnet/).
You need to install ESPnet following [ESPnet installation](https://espnet.github.io/espnet/installation.html).
Our Pytorch version is `2.2.0`, and you additionally need `triton==2.2.0` and `causal-conv1d==1.2.0.post2`.

## Training and Evaluation
Our decoder-only approach in ASR task is implemented as [asr2d](egs2/librispeech_100/asr2d/) for LibriSpeech 100h, which relies on [asr2](https://espnet.github.io/espnet/recipe/asr2.html) and [lm1](https://espnet.github.io/espnet/recipe/lm1.html) in the official repository.
We support MADEON as `lm: mamba` as you can see [here](egs2/librispeech_100/asr2d/conf/train_mamba.yaml).
Both parallel and serial versions of MADEON with speech prefixing (NADEON-SP) are also supported.
Please refer to the configurations.

### 1. Data preparation
- You will call Stages 1 to 6 of [asr2.sh](egs2/TEMPLATE/asr2/asr2.sh) to prepare data and tokenize speech via k-means clustering.
- Then, the dumped data will be converted to the `asr2d` format.
```
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
```

### 2. Training and evaluation of decoder-only model
- Stage 5 of [lm.sh](egs2/TEMPLATE/lm1/lm.sh) will prepare a joint vocabulary for speech and text tokens by using SentencePiece.
- You will collect stats and train the model on Stages 6 and 7, respectively.
- Stage 9 performs both decoding and evaluation, where the results are summarized in `RESULTS.md` similar to `asr1` and `asr2`.


```
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
```

## Licenses
This repository is based on a fork of [ESPnet](https://github.com/espnet/espnet).
All files except as noted below are with [Apache 2.0 license](LICENSE).

The following files:
- [espnet2/asr/state_spaces/mamba/mamba_mha.py](espnet2/asr/state_spaces/mamba/mamba_mha.py)
- [espnet2/asr/state_spaces/mamba/mamba.py](espnet2/asr/state_spaces/mamba/mamba.py)
- [espnet2/asr/state_spaces/mamba/mamba2.py](espnet2/asr/state_spaces/mamba/mamba2.py)
- [espnet2/asr/state_spaces/mamba/serialbimamba.py](espnet2/asr/state_spaces/mamba/serialbimamba.py)
- [espnet2/asr/state_spaces/ops/selective_scan_interface.py](espnet2/asr/state_spaces/ops/selective_scan_interface.py)

were adopted from [Mamba](https://github.com/state-spaces/mamba) and [Vision Mamba](https://github.com/hustvl/Vim/tree/main) (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md)) and modified for MADEON.

The files in 
- [espnet2/asr/state_spaces/distributed](espnet2/asr/state_spaces/distributed)
- [espnet2/asr/state_spaces/ops/triton](espnet2/asr/state_spaces/ops/triton/)

were copied without modification from [Mamba](https://github.com/state-spaces/mamba) (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md)).