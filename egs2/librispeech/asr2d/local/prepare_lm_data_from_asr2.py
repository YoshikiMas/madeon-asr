import argparse
import os
from pathlib import Path


def read_text(text: Path):
    uttid2text = {}
    with text.open("r") as fp:
        for line in fp.readlines():
            line = line.strip().split()
            uttid2text[line[0]] = " ".join(line[1:])
    return uttid2text


def prepare_asr(
    src_text: Path,
    tgt_text: Path,
    out_text: Path,
    start_speech_token="<startofspeech>",
    generate_text_token="<generatetext>",
):
    uttid2text = read_text(tgt_text)
    uttid2token = read_text(src_text)
    assert len(uttid2text) == len(uttid2token), (len(uttid2text), len(uttid2token))
    res = []
    for uttid, text in uttid2text.items():
        token = uttid2token[uttid].split()

        uttid = f"asr_{uttid}"
        text = text.lower()
        token = "".join(token)

        res.append(f"{uttid} {start_speech_token}{token}{generate_text_token} {text}")

    print("Creating asr: ", out_text)
    with (out_text).open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_feats",
        type=str,
        help="Directory for saving lm_text",
    )
    parser.add_argument(
        "--asr2_data_feats",
        type=str,
        help="Directory for original ASR2 source and target texts",
    )
    parser.add_argument(
        "--dset",
        type=str,
        help="",
    )
    parser.add_argument(
        "--src_case",
        type=str,
        help="",
        default="rm",
    )
    parser.add_argument(
        "--tgt_case",
        type=str,
        help="",
        default="ts",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        help="",
        default="wavlm_large_21_km2000",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        help="",
        default="en"
    )
    parser.add_argument(
        "--start_text_token",
        type=str,
        help="Token to denote start of text as condition.",
        default="<startoftext>",
    )
    parser.add_argument(
        "--generate_text_token",
        type=str,
        help="Token to denote generate text.",
        default="<generatetext>",
    )
    parser.add_argument(
        "--start_speech_token",
        type=str,
        help="Token to denote start of speech as condition.",
        default="<startofspeech>",
    )
    parser.add_argument(
        "--generate_speech_token",
        type=str,
        help="Token to denote generate speech.",
        default="<generatespeech>",
    )

    args = parser.parse_args()

    asr2 = Path(args.asr2_data_feats)
    asr2_src_text = asr2.joinpath(args.dset, f"text.{args.src_case}.{args.src_lang}")
    asr2_tgt_text = asr2.joinpath(args.dset, f"text.{args.tgt_case}.{args.tgt_lang}")
    lm1 = Path(args.data_feats)
    lm1_text = lm1.joinpath(args.dset, "lm_text")

    os.makedirs(lm1_text.parent, exist_ok=True)
    prepare_asr(
        src_text=asr2_src_text,
        tgt_text=asr2_tgt_text,
        out_text=lm1_text,
        start_speech_token=args.start_speech_token,
        generate_text_token=args.generate_text_token,
    )

    os.makedirs("data", exist_ok=True)
    nlsyms = Path("data").joinpath("nlsyms.txt")
    with (nlsyms).open("w") as fp:
        fp.write(
            "{}\n{}\n{}\n{}\n".format(
                args.start_text_token,
                args.generate_text_token,
                args.start_speech_token,
                args.generate_speech_token,
            )
        )


