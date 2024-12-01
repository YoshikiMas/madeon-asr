import argparse
from pathlib import Path


def prepare_asr_tts(input, output, prefix, sos):
    """Only keep the condition but remove the true target."""

    with open(input, "r") as fin, open(output, "w") as fout:
        for line in fin.readlines():
            if line.startswith(prefix):
                fout.write(line.split(sos)[0] + sos + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_feats",
        type=str,
        help="Directory for saving lm_text",
    )
    parser.add_argument(
        "--dset",
        type=str,
        help="",
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
    in_file = f"{args.data_feats}/{args.dset}/lm_text"
    out_file = f"{args.data_feats}/{args.dset}/text.asr"
    print(in_file, out_file)
    prepare_asr_tts(
        in_file, out_file, "asr_", args.generate_text_token
    )
