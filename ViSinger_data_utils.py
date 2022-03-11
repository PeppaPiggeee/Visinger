from copy import deepcopy
import glob
import os
'''
import librosa
from scipy.interpolate import interp1d
from textgrid import TextGrid
from utils import librosa_pad_lr
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import torch
from text import text_to_sequence, cleaned_text_to_sequence
from data_gen.tts.data_gen_utils import get_mel2ph, get_pitch, build_phone_encoder, is_sil_phoneme
import json
from utils.text_encoder import TokenTextEncoder
'''


class MetaDataProcessor():
    def __init__(self, dataset):
        self.wav_fns = sorted(glob.glob(f'{dataset}/0514#女_2_4*/*_wf0.wav'))
        self.train_wav_fns, self.test_wav_fns = self.split_train_test_set()
        self.train_meta_fn = f'filelists/xiaoma_audio_text_train_filelist.txt'
        self.test_meta_fn = f'filelists/xiaoma_audio_text_test_filelist.txt'
        self.train_other_fn = f'filelists/xiaoma_midi_duration_train_filelist.txt'
        self.test_other_fn = f'filelists/xiaoma_midi_duration_test_filelist.txt'
        self.train_meta_fn_ = f'filelists/xiaoma_audio_text_train_filelist_.txt.cleaned'
        self.test_meta_fn_ = f'filelists/xiaoma_audio_text_test_filelist_.txt.cleaned'



    def split_train_test_set(self):
        item_names = deepcopy(self.wav_fns)
        test_prefixes = ['0514#女_2_4-说散就散', '0514#女_2_4-隐形的翅膀']



        test_wav_fns = [x for x in item_names if any([ts in x for ts in test_prefixes])]
        train_wav_fns = [x for x in item_names if x not in set(test_wav_fns)]
        return train_wav_fns, test_wav_fns


    def get_files(self, prefix):
        if prefix == 'train':
            wav_fns = self.train_wav_fns
        else:
            wav_fns = self.test_wav_fns
        files = []
        for wav_fn in wav_fns:
            dir_name = os.path.dirname(wav_fn)
            base_name = os.path.basename(wav_fn)
            raw_txt_fn = os.path.join(dir_name, base_name[:-8] + '.txt')
            ph_fn = os.path.join(dir_name, base_name[:-8] + '_ph.txt')
            midi_fn = os.path.join(dir_name, base_name[:-8] + '.mid')
            f0_fn = os.path.join(dir_name, base_name[:-8] + '_f0.npy')
            tg_fn = os.path.join(dir_name, base_name[:-8] + '.TextGrid')

            files.append(
                {'wav_fn': wav_fn, 'txt_fn': raw_txt_fn, 'ph_fn': ph_fn, 'midi_fn': midi_fn, 'f0_fn': f0_fn, 'tg_fn': tg_fn})
        return files

    def load_txt(self, txt_fn):
        with open(txt_fn, 'r', encoding='utf-8') as f:
            txt = f.readline()
        return txt

    def process(self, prefix):
        fns = self.get_files(prefix)
        items = []
        others = []
        for fn in fns:
            txt_fn = fn['txt_fn']
            wav_fn = fn['wav_fn']
            ph_fn = fn['ph_fn']
            midi_fn = fn['midi_fn']
            f0_fn = fn['f0_fn']
            tg_fn = fn['tg_fn']


            txt = self.load_txt(ph_fn)
            item = wav_fn + "|" + txt + "\n"
            items.append(item)
            #other = midi_fn + "|" + f0_fn + "|" + tg_fn + "\n"
            #others.append(other)
        if prefix == 'train':
            meta_fn = self.train_meta_fn_
            #other_fn = self.train_other_fn
        else:
            meta_fn = self.test_meta_fn_
            #other_fn = self.test_other_fn
        file_handle = open(meta_fn, mode='w')
        file_handle.writelines(items)
        file_handle.close()
        #file_handle = open(other_fn, mode='w')
        #file_handle.writelines(others)
        #file_handle.close()



'''

def librosa_wav2spec(wav_path,
                     fft_size=1024,
                     hop_size=256,
                     win_length=1024,
                     window="hann",
                     num_mels=80,
                     fmin=80,
                     fmax=-1,
                     eps=1e-6,
                     sample_rate=22050):
    wav, _ = librosa.core.load(wav_path, sr=sample_rate)

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="constant")
    linear_spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)

    # calculate mel spec
    mel = mel_basis @ linear_spc
    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]

    # log linear spec
    linear_spc = np.log10(np.maximum(eps, linear_spc))
    return wav, mel, linear_spc
    #return {'wav': wav, 'mel': mel.T, 'linear': linear_spc.T}

def _word_encoder(dict_dir):
    fn = f"{dict_dir}/word_set.json"

    word_set = json.load(open(fn, 'r'))
    print("| Load word set. Size: ", len(word_set), word_set[:10])
    return TokenTextEncoder(None, vocab_list=word_set, replace_oov='<UNK>')



class VISingerLoader(torch.utils.data.Dataset):
    def __init__(self, dataset, hparams):
        self.wav_fns = sorted(glob.glob(f'{dataset}/0514#女_2_4*/*_wf0.wav'))
        self.files = self.getFiles()
        self.hp = hparams




    def getFiles(self):
        files = []
        for wav_fn in self.wav_fns:
            dir_name = os.path.dirname(wav_fn)
            base_name = os.path.basename(wav_fn)
            raw_txt_fn = os.path.join(dir_name, base_name[:-8] + '.txt')
            ph_fn = os.path.join(dir_name, base_name[:-8] + '_ph.txt')
            midi_fn = os.path.join(dir_name, base_name[:-8] + '.mid')
            f0_fn = os.path.join(dir_name, base_name[:-8] + '_f0.npy')
            tg_fn = os.path.join(dir_name, base_name[:-8] + '.TextGrid')
            files.append({'wav_fn':wav_fn, 'raw_txt_fn': raw_txt_fn, 'ph_fn': ph_fn, 'midi_fn': midi_fn, 'f0_fn': f0_fn, 'tg_fn': tg_fn})
        print(len(self.wav_fns))
        return files

    def get_wav_spec(self, index, hp):
        wav_fn = self.files[index]['wav_fn']
        wav, _ = librosa.core.load(wav_fn, sr=hp['sample_rate'])
        x_stft = librosa.stft(wav, n_fft=hp['win_size'], hop_length=hp['hop_size'], win_length=hp['win_size'], pad_mode="constant")
        wav_spec = librosa_wav2spec(wav_fn, hp['win_size'], hp['hop_size'], hp['win_size'], sample_rate=hp['sample_rate'])
        return wav_spec

    def get_ph(self, index, hp):
        ph_fn = self.files[index]['ph_fn']
        with open(ph_fn, "r", encoding="utf-8") as f:
            ph = f.readline()

        ph_list = ph.split(' ')
        ph_encoder, word_encoder = build_phone_encoder()
        return ph_encoder.encode(ph)

    #def process_item(self, ph, duration, midi):


    def get_notes_f0(self, index, hp):
        midi_fn = self.files[index]['midi_fn']
        f0_fn = self.files[index]['f0_fn']
        pm = pretty_midi.PrettyMIDI(midi_fn)
        f0 = np.load(f0_fn)
        notes = np.zeros_like(f0[:, 0])
        for n in pm.instruments[0].notes:
            notes[int(n.start * 100):int(n.end * 100)] = librosa.midi_to_hz(n.pitch)
        f0 = [x[1] for x in f0]
        return notes, f0


    def __getitem__(self, index):
        item = {}
        item['wav'], item['spec'], item['linear'] = self.get_wav_spec(index, self.hp)
        item['ph'] = self.get_ph(index, self.hp)
        item['notes'], item['f0'] = self.get_notes_f0(index, self.hp)
        return item

'''

if __name__ == '__main__':
    #hp = {'win_size': 1024, 'hop_size': 256, 'sample_rate': 22050}
    dataset = '../../Data/raw/xiaoma/splits_22050'

    #ds = VISingerLoader(dataset, hp)
    #print(ds[0])
    processor = MetaDataProcessor(dataset)
    processor.process('train')
    processor.process('test')




