import argparse
import os
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, help="Input file with all training data"
    )
    parser.add_argument("-o", "--output", type=str, help="Output file for bpe training")
    parser.add_argument("--upsampling_factor", type=int, default=1)
    parser.add_argument("--generate_text_token", type=str, default="<generatetext>")

    args = parser.parse_args()
    os.makedirs(Path(args.output).parent, exist_ok=True)

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin.readlines():
            if line.startswith("asr") or line.startswith("tts"):
                tgt_txt = line.rstrip().split(args.generate_text_token)[-1]
                exapanded_tgt_txt = (args.upsampling_factor - 1) * tgt_txt
                fout.write(line.rstrip()+exapanded_tgt_txt+"\n")

    print("Successfully created BPE training file")
