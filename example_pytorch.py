from os.path import join
from pathlib import Path

from audio_loader.features.raw_audio import WindowedAudio
from audio_loader.features.mfcc import WindowedMFCC
from audio_loader.ground_truth.timit import TimitGroundTruth
from audio_loader.samplers.windowed import WindowedSampler
from audio_loader.dl_frontends.pytorch.fill_ram import get_pytorch_dataloader_fill_ram

timit_gt = TimitGroundTruth(join(Path.home(), "data/darpa-timit-acousticphonetic-continuous-speech"))
print("groundtruth loaded")

raw_feature_processor = WindowedAudio(1024, 512, 16000)
raw_sampler = WindowedSampler([raw_feature_processor], timit_gt, 8000, overlap=0.5)

raw_train_dataloader = get_pytorch_dataloader_fill_ram(raw_sampler, 32, "train")
raw_test_dataloader = get_pytorch_dataloader_fill_ram(raw_sampler, 32, "test")

print("raw audio done")

mfcc_feature_processor = WindowedMFCC(1024, 512, 16000, 20)
mfcc_sampler = WindowedSampler([mfcc_feature_processor], timit_gt, 8000, overlap=0.5)
mfcc_train_dataloader = get_pytorch_dataloader_fill_ram(mfcc_sampler, 32, "train")
mfcc_test_dataloader = get_pytorch_dataloader_fill_ram(mfcc_sampler, 32, "test")


print("mfcc done")
