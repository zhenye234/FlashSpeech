# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.append('/scratch/buildlam/speech_yz/Amphion')
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from processors.acoustic_extractor import cal_normalized_mel
from processors.acoustic_extractor import load_normalized
from models.base.base_dataset import (
    BaseOfflineCollator,
    BaseOfflineDataset,
    BaseTestDataset,
    BaseTestCollator,
)
from text import text_to_sequence
from text.cmudict import valid_symbols
from tqdm import tqdm
import pickle
import tgt

class NS2Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):
        assert isinstance(dataset, str)

        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)

        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        # train.json

        self.metafile_path = os.path.join(processed_data_dir, meta_file)

        self.metadata = self.get_metadata()

        self.cfg = cfg

        assert cfg.preprocess.use_mel == False
        if cfg.preprocess.use_mel:
            self.utt2melspec_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2melspec_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.melspec_dir,  # mel
                    utt_info["Singer"],
                    uid + ".npy",
                )

        assert cfg.preprocess.use_code == True
        if cfg.preprocess.use_code:
            self.utt2code_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2code_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.code_dir,  # code
                    utt_info["Singer"],
                    uid + ".npy",
                )

        assert cfg.preprocess.use_spkid == True
        if cfg.preprocess.use_spkid:
            self.utt2spkid = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2spkid[utt] = utt_info["Singer"]

        assert cfg.preprocess.use_pitch == True
        if cfg.preprocess.use_pitch:
            self.utt2pitch_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2pitch_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.pitch_dir,  # pitch
                    utt_info["Singer"],
                    uid + ".npy",
                )

        assert cfg.preprocess.use_duration == True
        if cfg.preprocess.use_duration:
            self.utt2duration_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2duration_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.duration_dir,  # duration
                    utt_info["Singer"],
                    uid + ".npy",
                )

        # assert cfg.preprocess.use_phone == True
        # if cfg.preprocess.use_phone:
        #     self.utt2phone = {}
        #     for utt_info in self.metadata:
        #         dataset = utt_info["Dataset"]
        #         uid = utt_info["Uid"]
        #         utt = "{}_{}".format(dataset, uid)

        #         self.utt2phone[utt] = utt_info["phones"]

        assert cfg.preprocess.use_len == True
        if cfg.preprocess.use_len:
            self.utt2len = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                # hop size 200
                self.utt2len[utt] = utt_info['Duration']*80 #utt_info["num_frames"]

        # for cross reference
        if cfg.preprocess.use_cross_reference:
            self.spkid2utt = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                spkid = utt_info["Singer"]
                if spkid not in self.spkid2utt:
                    self.spkid2utt[spkid] = []
                self.spkid2utt[spkid].append(utt)

        # get phone to id / id to phone map
        self.phone2id, self.id2phone = self.get_phone_map()

        self.all_num_frames = []
        for i in range(len(self.metadata)):
            # self.all_num_frames.append(self.metadata[i]["num_frames"])
            self.all_num_frames.append(self.metadata[i]["Duration"]*80)
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )
        self.sampling_rate=16000
        self.hop_length=200

        # self.metadata_2=
        if not is_valid:
            with open('/scratch/buildlam/speech_yz/Amphion/data/libritts/processed_metadata.pkl', 'rb') as file: #processed_metadata
                self.metadata2 = pickle.load(file)
        else:
            with open('/scratch/buildlam/speech_yz/Amphion/data/libritts/processed_metadata_valid.pkl', 'rb') as file:
                self.metadata2 = pickle.load(file)             
             
        c=1

    def __len__(self):
        return len(self.metadata)

    def get_dataset_name(self):
        return self.metadata[0]["Dataset"]

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("metadata len: ", len(metadata))

        return metadata

    def get_phone_map(self):
        symbols = valid_symbols + ["sp", "spn", "sil"] + ["<s>", "</s>"]+["<br>"]
        phone2id = {s: i for i, s in enumerate(symbols)}
        id2phone = {i: s for s, i in phone2id.items()}
        return phone2id, id2phone
 
    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            if phones == []:
                if p in sil_phones:
 
                    phones.append("<s>")
                else:
 
                    phones.append("<s>")
                    durations.append(int(0))  
                    phones.append(p)
            else:
                if p not in sil_phones:
                    # For ordinary phones
                    phones.append(p)
                    # end_time = e
                    # end_idx = len(phones)
                else:
                    # For silent phones
                    phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        # phones = phones[:end_idx]
        # durations = durations[:end_idx]
        if phones[-1] in sil_phones:
            phones[-1] = "</s>"
        else:
            phones.append("</s>")
            durations.append(int(0))
        return phones, durations, start_time, end_time
         

    def load_metadata(self):
        processed_meta_path = os.path.join(os.path.dirname(self.metafile_path), 'processed_metadata.pkl')
        if os.path.exists(processed_meta_path):
            print(f"Loading preprocessed metadata from {processed_meta_path}")
            with open(processed_meta_path, 'rb') as file:
                metadata = pickle.load(file)
        else:
            print("Preprocessed metadata not found, processing data...")
            metadata = self.process_metadata()
            # Optionally save the processed metadata for future runs
            with open(processed_meta_path, 'wb') as file:
                pickle.dump(metadata, file)
        return metadata


    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()

        # if self.cfg.preprocess.read_metadata:
        #     metadata_uid_path = os.path.join(
        #         self.cfg.preprocess.processed_dir,
        #         self.cfg.preprocess.metadata_dir,
        #         dataset,
        #         # utt_info["Singer"],
        #         uid + ".pkl",
        #     )
        #     with open(metadata_uid_path, "rb") as f:
        #         metadata_uid = pickle.load(f)
        #     # code
        #     code = metadata_uid["code"]
        #     # frame_nums
        #     frame_nums = code.shape[1]
        #     # pitch
        #     pitch = metadata_uid["pitch"]
        #     # duration
        #     duration = metadata_uid["duration"]
        #     # phone_id
        #     phone_id = np.array(
        #         [
        #             *map(
        #                 self.phone2id.get,
        #                 self.utt2phone[utt].replace("{", "").replace("}", "").split(),
        #             )
        #         ]
        #     )

        # else:
 
            # path_formatted_uid = uid.replace('#', '/')
 
            # path_code = os.path.join('/scratch/buildlam/speech_yz/LibriTTS-codes3',f"{path_formatted_uid}.wav.npy")
            # # /aifs4su/data/zheny/Flashspeech/LibriTTS/LibriTTS/train-other-500/2834/132497/2834_132497_000027_000007.wav
            # code = np.load(path_code)
            # frame_nums = code.shape[1]
            # # pitch
            # pitch_root_path = '/scratch/buildlam/speech_yz/LibriTTS_pitch'
            # pitch_path = os.path.join(pitch_root_path, f"{path_formatted_uid}.f0.npy")
            # pitch = np.load(pitch_path)
 
            # textgrid_path = os.path.join('/scratch/buildlam/speech_yz/LibriTTS_text_grid_11',f"{path_formatted_uid}.TextGrid")
            # textgrid = tgt.io.read_textgrid(textgrid_path)
            # phone, duration, _, _ = self.get_alignment(
            #     textgrid.get_tier_by_name("phones")
            # )
            # phone_id = np.array(
            #     [
            #         *map(
            #             self.phone2id.get,
            #             phone,
            #         )
            #     ]
            # )
 
        # code, pitch, duration, phone_id, frame_nums = self.align_length(
        #     code, pitch, duration, phone_id, frame_nums
        # )


        code = self.metadata2[index]['code'][0]#.numpy()
        pitch = self.metadata2[index]['pitch'][0]#.numpy()
        duration = self.metadata2[index]['duration'][0]#.numpy()
        phone_id = self.metadata2[index]['phone_id'][0]#.numpy()
        frame_nums = self.metadata2[index]['frame_nums'][0]#.numpy()




        # single_feature.update(
        #     {
        #         "code": code,
        #         "frame_nums": frame_nums,
        #         "pitch": pitch,
        #         "duration": duration,
        #         "phone_id": phone_id,
        #         # "phone_id_frame": phone_id_frame,
        #         # "ref_code": ref_code,
        #         # "ref_frame_nums": ref_frame_nums,
        #         # "ref_pitch": ref_pitch,
        #         # "ref_duration": ref_duration,
        #         # "ref_phone_id": ref_phone_id,
        #         # "ref_phone_id_frame": ref_phone_id_frame,
        #         # "spkid": spkid,
        #     }
        # )

        # return single_feature


        # spkid
        spkid = self.utt2spkid[utt]

        # get target and reference
        out = self.get_target_and_reference(code, pitch, duration, phone_id, frame_nums)
        code, ref_code = out["code"], out["ref_code"]
        pitch, ref_pitch = out["pitch"], out["ref_pitch"]
        duration, ref_duration = out["duration"], out["ref_duration"]
        phone_id, ref_phone_id = out["phone_id"], out["ref_phone_id"]
        frame_nums, ref_frame_nums = out["frame_nums"], out["ref_frame_nums"]

        # phone_id_frame
        assert len(phone_id) == len(duration)
        phone_id_frame = []
        try:
            for i in range(len(phone_id)):
                phone_id_frame.extend([phone_id[i] for _ in range(duration[i])])
            phone_id_frame = np.array(phone_id_frame)
        except Exception as e:
            print(e)
            print(phone_id)
        # ref_phone_id_frame
        assert len(ref_phone_id) == len(ref_duration)
        ref_phone_id_frame = []
        for i in range(len(ref_phone_id)):
            ref_phone_id_frame.extend([ref_phone_id[i] for _ in range(ref_duration[i])])
        ref_phone_id_frame = np.array(ref_phone_id_frame)

        single_feature.update(
            {
                "code": code,
                "frame_nums": frame_nums,
                "pitch": pitch,
                "duration": duration,
                "phone_id": phone_id,
                "phone_id_frame": phone_id_frame,
                "ref_code": ref_code,
                "ref_frame_nums": ref_frame_nums,
                "ref_pitch": ref_pitch,
                "ref_duration": ref_duration,
                "ref_phone_id": ref_phone_id,
                "ref_phone_id_frame": ref_phone_id_frame,
                "spkid": spkid,
            }
        )

        return single_feature

    def get_num_frames(self, index):
        utt_info = self.metadata[index]
        # return utt_info["num_frames"]
        return utt_info["Duration"]*80

    def align_length(self, code, pitch, duration, phone_id, frame_nums):
        # aligh lenght of code, pitch, duration, phone_id, and frame nums
        code_len = code.shape[1]
        pitch_len = len(pitch)
        dur_sum = sum(duration)
        min_len = min(code_len, dur_sum)
        code = code[:, :min_len]
        if pitch_len >= min_len:
            pitch = pitch[:min_len]
        else:
            pitch = np.pad(pitch, (0, min_len - pitch_len), mode="edge")
        frame_nums = min_len
        if dur_sum > min_len:
            assert (duration[-1] - (dur_sum - min_len)) >= 0
            duration[-1] = duration[-1] - (dur_sum - min_len)
            assert duration[-1] >= 0

        return code, pitch, duration, phone_id, frame_nums

    # def get_target_and_reference(self, code, pitch, duration, phone_id, frame_nums):
    #     phone_nums = len(phone_id)
    #     clip_phone_nums = np.random.randint(
    #         int(phone_nums * 0.1), int(phone_nums * 0.5) + 1
    #     )
    #     clip_phone_nums = max(clip_phone_nums, 1)
    #     assert clip_phone_nums < phone_nums and clip_phone_nums >= 1
    #     if self.cfg.preprocess.clip_mode == "mid":
    #         start_idx = np.random.randint(0, phone_nums - clip_phone_nums)
    #     elif self.cfg.preprocess.clip_mode == "start":
    #         if duration[0] == 0 and clip_phone_nums == 1:
    #             start_idx = 1
    #         else:
    #             start_idx = 0
    #     else:
    #         assert self.cfg.preprocess.clip_mode in ["mid", "start"]
    #     end_idx = start_idx + clip_phone_nums
    #     start_frames = sum(duration[:start_idx])
    #     end_frames = sum(duration[:end_idx])

    #     new_code = np.concatenate(
    #         (code[:, :start_frames], code[:, end_frames:]), axis=1
    #     )
    #     ref_code = code[:, start_frames:end_frames]

    #     new_pitch = np.append(pitch[:start_frames], pitch[end_frames:])
    #     ref_pitch = pitch[start_frames:end_frames]

    #     new_duration = np.append(duration[:start_idx], duration[end_idx:])
    #     ref_duration = duration[start_idx:end_idx]

    #     new_phone_id = np.append(phone_id[:start_idx], phone_id[end_idx:])
    #     ref_phone_id = phone_id[start_idx:end_idx]

    #     new_frame_nums = frame_nums - (end_frames - start_frames)
    #     ref_frame_nums = end_frames - start_frames

    #     return {
    #         "code": new_code,
    #         "ref_code": ref_code,
    #         "pitch": new_pitch,
    #         "ref_pitch": ref_pitch,
    #         "duration": new_duration.astype(np.int64),
    #         "ref_duration": np.array(ref_duration) .astype(np.int64),
    #         "phone_id": new_phone_id,
    #         "ref_phone_id": ref_phone_id,
    #         "frame_nums": new_frame_nums,
    #         "ref_frame_nums": ref_frame_nums,
    #     }
    def get_target_and_reference(self, code, pitch, duration, phone_id, frame_nums):
        phone_nums = len(phone_id)
        clip_phone_nums = torch.randint(int(phone_nums * 0.1), int(phone_nums * 0.5) + 1, (1,)).item()
        clip_phone_nums = max(clip_phone_nums, 1)
        assert clip_phone_nums < phone_nums and clip_phone_nums >= 1
        
        if self.cfg.preprocess.clip_mode == "mid":
            start_idx = torch.randint(0, phone_nums - clip_phone_nums, (1,)).item()
        elif self.cfg.preprocess.clip_mode == "start":
            if duration[0] == 0 and clip_phone_nums == 1:
                start_idx = 1
            else:
                start_idx = 0
        else:
            assert self.cfg.preprocess.clip_mode in ["mid", "start"]
        
        end_idx = start_idx + clip_phone_nums
        start_frames = duration[:start_idx].sum().item()
        end_frames = duration[:end_idx].sum().item()

        new_code = torch.cat((code[:, :start_frames], code[:, end_frames:]), dim=1)
        ref_code = code[:, start_frames:end_frames]

        new_pitch = torch.cat((pitch[:start_frames], pitch[end_frames:]))
        ref_pitch = pitch[start_frames:end_frames]

        new_duration = torch.cat((duration[:start_idx], duration[end_idx:]))
        ref_duration = duration[start_idx:end_idx]

        new_phone_id = torch.cat((phone_id[:start_idx], phone_id[end_idx:]))
        ref_phone_id = phone_id[start_idx:end_idx]

        new_frame_nums = frame_nums - (end_frames - start_frames)
        ref_frame_nums = end_frames - start_frames

        return {
            "code": new_code,
            "ref_code": ref_code,
            "pitch": new_pitch,
            "ref_pitch": ref_pitch,
            "duration": new_duration,
            "ref_duration": ref_duration,
            "phone_id": new_phone_id,
            "ref_phone_id": ref_phone_id,
            "frame_nums": new_frame_nums,
            "ref_frame_nums": ref_frame_nums,
        }


class NS2Collator(BaseOfflineCollator):
    def __init__(self, cfg):
        BaseOfflineCollator.__init__(self, cfg)
        self.padding_value = 0
    def __call__(self, batch):
        packed_batch_features = dict()

        # code: (B, 16, T)
        # frame_nums: (B,)   not used
        # pitch: (B, T)
        # duration: (B, N)
        # phone_id: (B, N)
        # phone_id_frame: (B, T)
        # ref_code: (B, 16, T')
        # ref_frame_nums: (B,)   not used
        # ref_pitch: (B, T)   not used
        # ref_duration: (B, N')   not used
        # ref_phone_id: (B, N')   not used
        # ref_phone_frame: (B, T')   not used
        # spkid: (B,)   not used
        # phone_mask: (B, N)
        # mask: (B, T)
        # ref_mask: (B, T')

        for key in batch[0].keys():
            if key == "phone_id":
                phone_ids = [torch.LongTensor(b["phone_id"]) for b in batch]
                phone_masks = [torch.ones(len(b["phone_id"])) for b in batch]
                packed_batch_features["phone_id"] = pad_sequence(
                    phone_ids,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["phone_mask"] = pad_sequence(
                    phone_masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "phone_id_frame":
                phone_id_frames = [torch.LongTensor(b["phone_id_frame"]) for b in batch]
                masks = [torch.ones(len(b["phone_id_frame"])) for b in batch]
                packed_batch_features["phone_id_frame"] = pad_sequence(
                    phone_id_frames,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["mask"] = pad_sequence(
                    masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "ref_code":
                # ref_codes = [
                #     torch.from_numpy(b["ref_code"]).transpose(0, 1) for b in batch
                # ]
                ref_codes = [
                     b["ref_code"].transpose(0, 1) for b in batch
                ]
                ref_masks = [torch.ones(max(b["ref_code"].shape[1], 1)) for b in batch]
                packed_batch_features["ref_code"] = pad_sequence(
                    ref_codes,
                    batch_first=True,
                    padding_value=0,
                ).transpose(1, 2)
                packed_batch_features["ref_mask"] = pad_sequence(
                    ref_masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "code":
                # codes = [torch.from_numpy(b["code"]).transpose(0, 1) for b in batch]
                codes = [b["code"].transpose(0, 1) for b in batch]
                masks = [torch.ones(max(b["code"].shape[1], 1)) for b in batch]
                packed_batch_features["code"] = pad_sequence(
                    codes,
                    batch_first=True,
                    padding_value=0,
                ).transpose(1, 2)
                packed_batch_features["mask"] = pad_sequence(
                    masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "pitch":
                # values = [torch.from_numpy(b[key]) for b in batch]
                values = [b[key] for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=50.0
                )
            elif key == "duration":
                # values = [torch.from_numpy(b[key]) for b in batch]
                values = [b[key] for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )
            elif key == "frame_nums":
                packed_batch_features["frame_nums"] = torch.LongTensor(
                    [b["frame_nums"] for b in batch]
                )
            elif key == "ref_frame_nums":
                packed_batch_features["ref_frame_nums"] = torch.LongTensor(
                    [b["ref_frame_nums"] for b in batch]
                )
            else:
                pass

        return packed_batch_features
    # def __call__(self, batch):
    #     packed_batch_features = {}

    #     for key in batch[0].keys():
    #         items = [b[key] for b in batch]

    #         # Check if items are tensors, if not, convert them
    #         if not all(isinstance(x, torch.Tensor) for x in items):
    #             items = [torch.tensor(x, dtype=torch.int64 if isinstance(x, int) else torch.float32) for x in items]

    #         if any(isinstance(x, torch.Tensor) for x in items):
    #             try:
    #                 # Special handling for 'frame_nums' or similar non-sequence single value keys
    #                 if key == 'frame_nums' or key.endswith('nums'):  # assuming these hold scalar values
    #                     packed_batch_features[key] = torch.tensor(items, dtype=torch.int64)
    #                 elif 'code' in key or 'pitch' in key or 'ref_code' in key:
    #                     # Ensure proper dimensions for codes and pitches
    #                     items = [x.unsqueeze(0) if x.dim() == 1 else x for x in items]
    #                     packed_batch_features[key] = pad_sequence(items, batch_first=True, padding_value=self.padding_value)
    #                 else:
    #                     # Default handling for other sequence data
    #                     packed_batch_features[key] = pad_sequence(items, batch_first=True, padding_value=self.padding_value)
    #             except Exception as e:
    #                 print(f"Error processing key {key}: {e}")
    #         else:
    #             # Handle non-tensor but stackable data
    #             packed_batch_features[key] = torch.stack(items)

    #     return packed_batch_features
def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


from utils.util import load_config

# def test_ns2_dataset():
#     cfg = load_config('egs/tts/NaturalSpeech2/exp_config.json')
#     dataset_name = 'libritts'
#     is_valid = False

#     # Create the dataset object
#     dataset = NS2Dataset(cfg, dataset_name, is_valid)

#     # Fetch a sample from the dataset
#     sample_index = 0
#     if len(dataset) > 0:
#         sample = dataset[sample_index]
#         print("Sample keys:", sample.keys())
#         print("Sample data for 'code':", sample['code'])
#         print("Sample 'frame_nums':", sample['frame_nums'])
#     else:
#         print("Dataset is empty.")

# # Run the test function
# test_ns2_dataset()
import torch
from torch.utils.data import DataLoader

def save_metadata(dataset, output_path):
    import pickle
    # Save the metadata using pickle
    with open(output_path, 'wb') as file:
        pickle.dump(dataset.metadata, file)
    print(f"Metadata saved to {output_path}")

def process_and_save_metadata(dataset, output_path):
    # Assuming NS2Dataset is already imported and properly configured
    # Initialize the DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # Batch size can be larger based on your RAM

    # Iterate over the dataset
    for i, data in enumerate(data_loader):
        print(f"Processing {i+1}/{len(dataset)}")
        c=1
        # if i>1000:
        #     break
        # Here we just loop through the dataset
        # If you need to update metadata, modify `data` and update `dataset.metadata` accordingly
        dataset.metadata[i]['code']=data['code']
        dataset.metadata[i]['pitch']=data['pitch']
        dataset.metadata[i]['duration']=data['duration']
        dataset.metadata[i]['phone_id']=data['phone_id']
        dataset.metadata[i]['frame_nums']=data['frame_nums']
    # After processing all data, save the metadata
    save_metadata(dataset, output_path)

if __name__ == "__main__":
    # Configuration for NS2Dataset
    cfg = load_config('egs/tts/NaturalSpeech2/exp_config.json')
    dataset_name = 'libritts'
    is_valid = True
    # Create the dataset
    dataset = NS2Dataset(cfg, dataset_name, is_valid)

    # Output path for the saved metadata
    output_path = '/scratch/buildlam/speech_yz/Amphion/data/libritts/processed_metadata_valid.pkl'

    # # Process the dataset and save the metadata
    # process_and_save_metadata(dataset, output_path)
#     # Fetch a sample from the dataset
    sample_index = 0
    if len(dataset) > 0:
        sample = dataset[sample_index]
    #     print("Sample keys:", sample.keys())
    #     print("Sample data for 'code':", sample['code'])
    #     print("Sample 'frame_nums':", sample['frame_nums'])
    # else:
    #     print("Dataset is empty.")