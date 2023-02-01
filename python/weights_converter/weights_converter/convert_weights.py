from tqdm import tqdm
import urllib.request
import numpy as np
import argparse
import warnings
import torch
import os
import h5py


def main():
    _MODELS = {
        "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
        "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
        "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
        "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
        "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
        "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
        "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
        "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
    }

    _COMPRESSIONS = ["gzip", "None"]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='')

    parser.add_argument("--model", default="tiny", choices=_MODELS, help="the name of the scale sizse for the Whisper model to convert")
    parser.add_argument("--compression", "-c", choices=_COMPRESSIONS, default="gzip", help="use compression filter or not")
    parser.add_argument("--local_path", "-l", help="get weights from local directory")
    parser.add_argument("--output_dir", help="directory to save the converted weights")


    args = parser.parse_args().__dict__
    model_name  = args.pop("model")
    local_path  = args.pop("local_path")
    compression = args.pop("compression")
    output_dir  = args["output_dir"]

    if local_path:
        weights = torch.load(local_path)

        if output_dir: 
            output_dir = os.path.join(output_dir, f"{os.path.split(local_path)[1][:-3]}.h5")
        else:
            output_dir = f".\{os.path.split(local_path)[1][:-3]}.h5"

        if os.path.exists(output_dir):
            warnings.warn("The weight file already exists in this directory. The file will be overwritten.")
    else:
        if output_dir:
            output_dir_pt = os.path.join(output_dir, f"{model_name}.pt")
        else:
            output_dir_pt = f".\{model_name}.pt"

        if os.path.exists(output_dir_pt):
            warnings.warn("The weight file already exists in this directory. The file will be overwritten.")

        with urllib.request.urlopen(_MODELS[model_name]) as source, open(output_dir_pt, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

        if output_dir:
            output_dir = os.path.join(output_dir, f"{model_name}.h5")
        else:
            output_dir = f".\{model_name}.h5"

        weights = torch.load(output_dir_pt)


    for key in weights['model_state_dict'].keys():

        cur_weight = weights['model_state_dict'][key]

        if key in ['encoder.conv2.weight', 'encoder.conv1.weight']:
            weights['model_state_dict'][key] = torch.permute(weights['model_state_dict'][key], (2, 1, 0))

        if len(cur_weight.shape) == 2:
            if key in ['decoder.positional_embedding', 'encoder.positional_embedding', 'decoder.token_embedding.weight']:
                continue

            weights['model_state_dict'][key] = torch.transpose(weights['model_state_dict'][key], 1, 0)

    with h5py.File(output_dir, 'w') as f:

        group_1 = f.create_group('dims')
        group_2 = f.create_group('model_state_dict')

        for key in weights['dims'].keys():
            group_1.create_dataset(f'{key}', data=weights['dims'][key])

        if compression != "None":
            for key in weights['model_state_dict'].keys():
                group_2.create_dataset(f'{key}', data=weights['model_state_dict'][key].numpy().astype(np.float32), compression=compression)
        else:
            for key in weights['model_state_dict'].keys():
                group_2.create_dataset(f'{key}', data=weights['model_state_dict'][key].numpy().astype(np.float32))

if __name__ == "__main__":
    main()