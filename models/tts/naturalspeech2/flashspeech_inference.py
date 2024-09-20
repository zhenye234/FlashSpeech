# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
import soundfile as sf
import numpy as np

# from models.tts.naturalspeech2.ns2 import NaturalSpeech2
from models.tts.naturalspeech2.flashspeech import FlashSpeech
from encodec import EncodecModel
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
        # self.codec = self.build_codec()

        self.symbols = valid_symbols + ["sp", "spn", "sil"] + ["<s>", "</s>"]
        self.phone2id = {s: i for i, s in enumerate(self.symbols)}
        self.id2phone = {i: s for s, i in self.phone2id.items()}

    def build_model(self):
        # model = NaturalSpeech2(self.cfg.model)
        model = FlashSpeech(self.cfg.model)
        print('build flashspeeh')
        # model.load_state_dict(
        #     torch.load(
        #         os.path.join(self.args.checkpoint_path, "pytorch_model.bin"),
        #         map_location="cpu",
        #     )
        # )
        # aa= model.load_state_dict(torch.load(self.args.checkpoint_path,map_location="cpu")['state_dict'],strict=False)
        ckpt = torch.load(self.args.checkpoint_path, map_location="cpu")
        state_dict = ckpt['state_dict']

        # 调整键名
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # 加载模型参数
        model.load_state_dict(new_state_dict,strict=False)
        model = model.to(self.args.device)
        return model

 

    def get_ref_code(self):
        ref_wav_path = self.args.ref_audio
        ref_wav, sr = torchaudio.load(ref_wav_path)
        ref_wav = convert_audio(
            ref_wav, sr, 16000, 1
        )
        ref_wav = ref_wav.unsqueeze(0).to(device=self.args.device)

        with torch.no_grad():
            # encoded_frames = self.codec.encode(ref_wav)
            encoded_frames = self.model.soundstream.encode(ref_wav,None)
            # ref_code = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
            ref_code = encoded_frames[:,0,:]
        # print(ref_code.shape)

        # ref_mask = torch.ones(ref_code.shape[0], ref_code.shape[-1]).to(ref_code.device)
        ref_mask = torch.ones(1, ref_code.shape[-1]).to(ref_code.device)
        # print(ref_mask.shape)

        return ref_code, ref_mask

    def inference(self):
        ref_code, ref_mask = self.get_ref_code()

        lexicon = read_lexicon(self.cfg.preprocess.lexicon_path)
        phone_seq = preprocess_english(self.args.text, lexicon)
        phone_seq="<s> "+phone_seq+" </s>"
        print(phone_seq)

        phone_id = np.array(
            [
                *map(
                    self.phone2id.get,
                    phone_seq.replace("{", "").replace("}", "").split(),
                )
            ]
        )
        phone_id = torch.from_numpy(phone_id).unsqueeze(0).to(device=self.args.device)
        print(phone_id)
        print('inference_step',self.args.inference_step)
        x0, prior_out = self.model.inference(
            ref_code, phone_id, ref_mask, self.args.inference_step
        )
        print(prior_out["dur_pred"])
        print(prior_out["dur_pred_round"])
        print(torch.sum(prior_out["dur_pred_round"]))

        ref_wav = self.model.soundstream.decode(ref_code.unsqueeze(1)) #.transpose(0, 1))

        rec_wav = self.model.soundstream.decoder_2(x0*3)
        # ref_wav = self.codec.decoder(latent_ref)

        os.makedirs(self.args.output_dir, exist_ok=True)

        sf.write(
            "{}/{}.wav".format(
                self.args.output_dir, self.args.text.replace(" ", "_", 100)
            ),
            rec_wav[0, 0].detach().cpu().numpy(),
            samplerate=16000,
        )

    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--ref_audio",
            type=str,
            default="",
            help="Reference audio path",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
        )
        parser.add_argument(
            "--inference_step",
            type=int,
            default=4,
            help="Total inference steps for the diffusion model",
        )



class FlashSpeechInference2:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args

        self.model = self.build_model()
        # self.codec = self.build_codec()

        self.symbols = valid_symbols + ["sp", "spn", "sil"] + ["<s>", "</s>"]
        self.phone2id = {s: i for i, s in enumerate(self.symbols)}
        self.id2phone = {i: s for s, i in self.phone2id.items()}

    def build_model(self):
        # model = NaturalSpeech2(self.cfg.model)
        model = FlashSpeech(self.cfg.model)
        print('build flashspeeh')
        # model.load_state_dict(
        #     torch.load(
        #         os.path.join(self.args.checkpoint_path, "pytorch_model.bin"),
        #         map_location="cpu",
        #     )
        # )
        aa= model.load_state_dict(torch.load(self.args.checkpoint_path,map_location="cpu")['state_dict'],strict=False)
        ckpt = torch.load(self.args.checkpoint_path, map_location="cpu")
        state_dict = ckpt['state_dict']

        # 调整键名
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # 加载模型参数
        model.load_state_dict(new_state_dict,strict=False)
        model = model.to(self.args.device)
        return model

 

    def get_ref_code(self):
        ref_wav_path = self.args.ref_audio
        ref_wav, sr = torchaudio.load(ref_wav_path)
        ref_wav = convert_audio(
            ref_wav, sr, 16000, 1
        )
        ref_wav = ref_wav.unsqueeze(0).to(device=self.args.device)

        with torch.no_grad():
            # encoded_frames = self.codec.encode(ref_wav)
            encoded_frames = self.model.soundstream.encode(ref_wav,None)
            # ref_code = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
            ref_code = encoded_frames[:,0,:]
        # print(ref_code.shape)

        # ref_mask = torch.ones(ref_code.shape[0], ref_code.shape[-1]).to(ref_code.device)
        ref_mask = torch.ones(1, ref_code.shape[-1]).to(ref_code.device)
        # print(ref_mask.shape)

        return ref_code, ref_mask

    def inference(self):
        ref_code, ref_mask = self.get_ref_code()

        import sys
        sys.path.append('/scratch/buildlam/speech_yz/new_duration_model/seamless_communication')
        from seamless_communication.models.aligner.alignment_extractor import AlignmentExtractor
 

        extractor = AlignmentExtractor(
            aligner_model_name_or_card="nar_t2u_aligner",
            unit_extractor_model_name_or_card="xlsr2_1b_v2",
            unit_extractor_output_layer=35,
            unit_extractor_kmeans_model_uri="https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
        )
        tokenized_text_ids =  extractor.alignment_model.alignment_frontend.tokenize_text(
                self.args.text, add_trailing_silence=True
            )
 
        # lexicon = read_lexicon(self.cfg.preprocess.lexicon_path)
        # phone_seq = preprocess_english(self.args.text, lexicon)
        # phone_seq="<s> "+phone_seq+" </s>"
        # print(phone_seq)

        # phone_id = np.array(
        #     [
        #         *map(
        #             self.phone2id.get,
        #             phone_seq.replace("{", "").replace("}", "").split(),
        #         )
        #     ]
        # )
        # phone_id = torch.from_numpy(phone_id).unsqueeze(0).to(device=self.args.device)
        # print(phone_id)
        phone_id = torch.tensor(tokenized_text_ids).unsqueeze(0).to(device=self.args.device)
        print('inference_step',self.args.inference_step)
        x0, prior_out = self.model.inference(
            ref_code, phone_id, ref_mask, self.args.inference_step
        )
        print(prior_out["dur_pred"])
        print(prior_out["dur_pred_round"])
        print(torch.sum(prior_out["dur_pred_round"]))

        ref_wav = self.model.soundstream.decode(ref_code.unsqueeze(1)) #.transpose(0, 1))

        rec_wav = self.model.soundstream.decoder_2(x0*3)
        # ref_wav = self.codec.decoder(latent_ref)

        os.makedirs(self.args.output_dir, exist_ok=True)

        sf.write(
            "{}/{}.wav".format(
                self.args.output_dir, self.args.text.replace(" ", "_", 100)
            ),
            rec_wav[0, 0].detach().cpu().numpy(),
            samplerate=16000,
        )

    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--ref_audio",
            type=str,
            default="",
            help="Reference audio path",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
        )
        parser.add_argument(
            "--inference_step",
            type=int,
            default=4,
            help="Total inference steps for the diffusion model",
        )
