# Modified FlashSpeech code to perform inference on LibriSpeech test-clean
# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.append('/scratch/buildlam/speech_yz/Amphion2')
import argparse
import os
import torch
import soundfile as sf
import numpy as np
import json
from tqdm import tqdm

# Assuming these imports are available in your project
from models.tts.naturalspeech2.flashspeech import FlashSpeech
from encodec.utils import convert_audio
from utils.util import load_config
from text import text_to_sequence
from text.cmudict import valid_symbols
from text.g2p import preprocess_english, read_lexicon
import torchaudio

class FlashSpeechInference:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args

        self.model = self.build_model()

        self.symbols = valid_symbols + ["sp", "spn", "sil"] + ["<s>", "</s>"]
        self.phone2id = {s: i for i, s in enumerate(self.symbols)}
        self.id2phone = {i: s for s, i in self.phone2id.items()}

        # Load lexicon
        self.lexicon = read_lexicon(self.cfg.preprocess.lexicon_path)

    def build_model(self):
        model = FlashSpeech(self.cfg.model)
        print('Building FlashSpeech model')

        ckpt = torch.load(self.args.checkpoint_path, map_location="cpu")
        state_dict = ckpt['state_dict']

        # Adjust key names
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # Load model parameters
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.args.device)
        model.eval()
        return model

    def get_ref_code(self, ref_wav_path):
        ref_wav, sr = torchaudio.load(ref_wav_path)
        ref_wav = convert_audio(
            ref_wav, sr, 16000, 1
        )
        ref_wav = ref_wav.unsqueeze(0).to(device=self.args.device)

        with torch.no_grad():
            encoded_frames = self.model.soundstream.encode(ref_wav, None)
            ref_code = encoded_frames[:, 0, :]
        ref_mask = torch.ones(1, ref_code.shape[-1]).to(ref_code.device)
        return ref_code, ref_mask

    def inference(self, text, ref_audio_path, output_wav_path):
        ref_code, ref_mask = self.get_ref_code(ref_audio_path)

        phone_seq = preprocess_english(text, self.lexicon)
        phone_seq = "<s> " + phone_seq + " </s>"
        print(f"Phoneme sequence: {phone_seq}")

        phone_id = np.array(
            [
                *map(
                    self.phone2id.get,
                    phone_seq.replace("{", "").replace("}", "").split(),
                )
            ]
        )
        phone_id = torch.from_numpy(phone_id).unsqueeze(0).to(device=self.args.device)
        print(f"Phone IDs: {phone_id}")
        print('Inference steps:', self.args.inference_step)

        x0, prior_out = self.model.inference(
            ref_code, phone_id, ref_mask, self.args.inference_step
        )

        print("Duration Prediction:", prior_out["dur_pred"])
        print("Rounded Duration Prediction:", prior_out["dur_pred_round"])
        print("Total Duration:", torch.sum(prior_out["dur_pred_round"]))

        rec_wav = self.model.soundstream.decoder_2(x0 * 3)

        sf.write(
            output_wav_path,
            rec_wav[0, 0].detach().cpu().numpy(),
            samplerate=16000,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='egs/tts/NaturalSpeech2/exp_config_s1.json', help="Path to config file")
    parser.add_argument("--checkpoint_path", type=str, default='/project/buildlam/zhenye/flashspeech_log/ns2_ict_normal_lignt_666_12node_smaller_lr_old_phone_s1_crop_mid_s2/epochepoch=40-stepstep=34727.ckpt', help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default='/scratch/buildlam/speech_yz/Amphion2/ckpts/output_all_mid3', help="Output directory for synthesized audio")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--inference_step", type=int, default=4, help="Total inference steps for the diffusion model")
    parser.add_argument("--input_json", type=str, default='/scratch/buildlam/speech_yz/dataset/librispeech/LibriSpeech/test.json', help="Path to input JSON file containing test cases")
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    inference_engine = FlashSpeechInference(args, cfg)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test cases from JSON file
    with open(args.input_json, 'r') as f:
        res = json.load(f)
    test_cases = res['test_cases']
    # Optionally, select every 10th sample as in the Valle code
    test_cases = [test_cases[i] for i in range(0, len(test_cases), 10)]

    for idx, case in tqdm(enumerate(test_cases), total=len(test_cases)):
        text = case['transcription_with_punc']
        ref_audio_path = case['reference_wav_path']
        output_wav_path = os.path.join(args.output_dir, f"{idx}.wav")
        inference_engine.inference(text, ref_audio_path, output_wav_path)

if __name__ == "__main__":
    main()




# class FlashSpeechInference:
#     def __init__(self, args, cfg):
#         self.cfg = cfg
#         self.args = args

#         self.model = self.build_model()

#         self.symbols = valid_symbols + ["sp", "spn", "sil"] + ["<s>", "</s>"]
#         self.phone2id = {s: i for i, s in enumerate(self.symbols)}
#         self.id2phone = {i: s for s, i in self.phone2id.items()}

#         # Load lexicon
#         self.lexicon = read_lexicon(self.cfg.preprocess.lexicon_path)

#     def build_model(self):
#         model = FlashSpeech(self.cfg.model)
#         print('Building FlashSpeech model')

#         ckpt = torch.load(self.args.checkpoint_path, map_location="cpu")
#         state_dict = ckpt['state_dict']

#         # Adjust key names
#         new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

#         # Load model parameters
#         model.load_state_dict(new_state_dict, strict=False)
#         model = model.to(self.args.device)
#         model.eval()
#         return model

#     def get_ref_code(self, ref_wav_path):
#         ref_wav, sr = torchaudio.load(ref_wav_path)
#         ref_wav = convert_audio(
#             ref_wav, sr, 16000, 1
#         )
#         ref_wav = ref_wav.unsqueeze(0).to(device=self.args.device)

#         with torch.no_grad():
#             encoded_frames = self.model.soundstream.encode(ref_wav, None)
#             ref_code = encoded_frames[:, 0, :]
#         ref_mask = torch.ones(1, ref_code.shape[-1]).to(ref_code.device)
#         return ref_code, ref_mask

#     def inference(self, text, ref_audio_path, output_wav_path):
#         ref_code, ref_mask = self.get_ref_code(ref_audio_path)

#         import re
#         import sys
#         sys.path.append('/scratch/buildlam/speech_yz/new_duration_model/seamless_communication')
#         from seamless_communication.models.aligner.alignment_extractor import AlignmentExtractor
 
#         text = re.sub(r"[^\w\s,.'\"]", "", text)
#         # 将所有字母转换为小写
#         text = text.lower()
#         extractor = AlignmentExtractor(
#             aligner_model_name_or_card="nar_t2u_aligner",
#             unit_extractor_model_name_or_card="xlsr2_1b_v2",
#             unit_extractor_output_layer=35,
#             unit_extractor_kmeans_model_uri="https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
#         )
#         tokenized_text_ids =  extractor.alignment_model.alignment_frontend.tokenize_text(
#                 text, add_trailing_silence=True
#             )
#         phone_id = torch.tensor(tokenized_text_ids).unsqueeze(0).to(device=self.args.device)

#         # phone_seq = preprocess_english(text, self.lexicon)
#         # phone_seq = "<s> " + phone_seq + " </s>"
#         # print(f"Phoneme sequence: {phone_seq}")

#         # phone_id = np.array(
#         #     [
#         #         *map(
#         #             self.phone2id.get,
#         #             phone_seq.replace("{", "").replace("}", "").split(),
#         #         )
#         #     ]
#         # )
#         # phone_id = torch.from_numpy(phone_id).unsqueeze(0).to(device=self.args.device)
#         # print(f"Phone IDs: {phone_id}")
#         print('Inference steps:', self.args.inference_step)

#         x0, prior_out = self.model.inference(
#             ref_code, phone_id, ref_mask, self.args.inference_step
#         )

#         print("Duration Prediction:", prior_out["dur_pred"])
#         print("Rounded Duration Prediction:", prior_out["dur_pred_round"])
#         print("Total Duration:", torch.sum(prior_out["dur_pred_round"]))

#         rec_wav = self.model.soundstream.decoder_2(x0 * 3)

#         sf.write(
#             output_wav_path,
#             rec_wav[0, 0].detach().cpu().numpy(),
#             samplerate=16000,
#         )

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config_path", type=str, default='egs/tts/NaturalSpeech2/exp_config.json', help="Path to config file")
#     parser.add_argument("--checkpoint_path", type=str, default='/project/buildlam/zhenye/flashspeech_log/ns2_ict_normal_lignt_666_2node_smaller_lr_new_phone2/epochepoch=399-stepstep=128000.ckpt', help="Path to model checkpoint")
#     parser.add_argument("--output_dir", type=str, default='/scratch/buildlam/speech_yz/Amphion2/ckpts/output_all_new_phone', help="Output directory for synthesized audio")
#     parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
#     parser.add_argument("--inference_step", type=int, default=4, help="Total inference steps for the diffusion model")
#     parser.add_argument("--input_json", type=str, default='/scratch/buildlam/speech_yz/dataset/librispeech/LibriSpeech/test.json', help="Path to input JSON file containing test cases")
#     args = parser.parse_args()

#     cfg = load_config(args.config_path)
#     inference_engine = FlashSpeechInference(args, cfg)

#     os.makedirs(args.output_dir, exist_ok=True)

#     # Load test cases from JSON file
#     with open(args.input_json, 'r') as f:
#         res = json.load(f)
#     test_cases = res['test_cases']
#     # Optionally, select every 10th sample as in the Valle code
#     test_cases = [test_cases[i] for i in range(0, len(test_cases), 10)]

#     for idx, case in tqdm(enumerate(test_cases), total=len(test_cases)):
#         text = case['transcription_with_punc']
#         ref_audio_path = case['reference_wav_path']
#         output_wav_path = os.path.join(args.output_dir, f"{idx}.wav")
#         inference_engine.inference(text, ref_audio_path, output_wav_path)

# if __name__ == "__main__":
#     main()



