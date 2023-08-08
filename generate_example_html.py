import argparse

parser = argparse.ArgumentParser(description='Generate example markdown file')
parser.add_argument('output', type=str, help='output html file')
parser.add_argument('directory', type=str, help='directory with the audio files')
# flag if we want to inluce the diffusion process gif
parser.add_argument('--gif', action='store_true', help='include diffusion process gif')

args = parser.parse_args()

import os
import glob

audio_files = glob.glob(os.path.join(args.directory, '*.wav'))

audio_dict = {}

for audio_file in audio_files:
    aid = "_".join(audio_file.split("_")[-3:]).split(".")[0]
    atype = audio_file.split("_")[0].split("/")[-1]
    if aid not in audio_dict:
        audio_dict[aid] = {}
    audio_dict[aid][atype] = audio_file.replace("examples/", "")

# same thing but for html
with open(args.output, "w") as f:
    f.write("<h1>TTS Audio</h1>\n")
    f.write("<p>The following TTS examples were generated using Teacher Forcing - a ground truth phone-level prosody representation was provided.</p>\n")
    f.write("<table>\n")
    f.write("<tr><th>Prompt</th><th>Ground Truth</th><th>TTS</th></tr>\n")
    for k, v in audio_dict.items():
        #f.write(f"<tr><td><img src='{v['prompt']}' /></td><td><img src='{v['gt']}' /></td><td><img src='{v['audio']}' /></td></tr>\n")
        # using audio tag instead of img tag
        f.write(f"<tr><td><audio controls><source src='{v['prompt']}' type='audio/wav'></audio></td><td><audio controls><source src='{v['gt']}' type='audio/wav'></audio></td><td><audio controls><source src='{v['audio']}' type='audio/wav'></audio></td></tr>\n")
    f.write("</table>\n")
    if args.gif:
        f.write("<h1>Diffusion Process</h1>\n")
        f.write("<details><summary>Diffusion Process Gif</summary>\n")
        gif_dir = args.directory.replace("examples/", "")
        f.write(f"<img src='{gif_dir}diffusion_process.gif' />\n")
        f.write("</details>\n")