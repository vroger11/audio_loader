# Audio Loader

Library designed to load audio batches (with features and ground truth) for deep learning libraries such as PyTorch or TensorFlow.

Its online documentation is [here](http://website.vincent-roger.fr/audio_loader/).

It is for now an early work that I do along my PhD (I use this library for my different projects).
Many things are missing but some are on their way like a documentation website.

It supports computations of features such as raw audio, MFCC and log spectrogram (using [librosa](https://librosa.github.io/librosa/)) more binding will be available in the future.
Also designed to ease the creation of new parsers for new datasets/challenges (supervised or not).
To do so, the library provides handy interfaces the users should follow.
Have a look into `audio_loader/ground_truth` package for more details.

For now, only a TIMIT like parser (with only phoneme ground truth support for now) and a C2SI-like parser are shared and only PyTorch is supported (TensorFlow support and other datasets are on my TODO list).

Feel free to add other libraries and/or challenges support with adequate tests.

## Install

### Miniconda

To install it you needs anaconda or miniconda installed then you to type:

```bash
conda env create -f environment.yml
conda activate audio_loader
```

All tests passed using this environment.

## Check if tests pass
For this project I used [pytest](https://docs.pytest.org/en/latest/) to write my tests. To try if every tests pass on your side, just type:

```bash
py.test tests/
```

If you want to improve mines feel free to help.
