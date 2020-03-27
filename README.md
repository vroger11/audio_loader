# Audio Loader
Library designed to load audio batches (with features and ground truth) for deep learning libraries such as PyTorch or TensorFlow.

It is for now an early work that I do alongside of my PhD (I use this library for my different projects).
Many things are missing but some are on their way like a documentation website.

It supports computations of features such as raw audio and MFCC (using [librosa](https://librosa.github.io/librosa/)) more binding will be available in the future.
It is designed to ease the creation of new parsers for new datasets/challenges (supervised or not).
To do so, the library provide handy interfaces the users should follow.
Have a look into `audio_loader/ground_truth` package for more details.

For now, only a TIMIT like parser (with only phoneme ground truth support for now) and a C2SI like parser are shared and only PyTorch is supported (TensorFlow support and other datasets are on my TODO list).

Feel free to add other libraries and/or challenges support with adequate tests.

# Miniconda installation
To install it you needs anaconda or miniconda installed and then you to type:

```bash
conda env create -f environment.yml
conda activate audio_loader
```

All tests passed using this environment.

# Library Class Diagram
`architecture.xmi` contains the class diagram and was released using [Umbrello project](https://umbrello.kde.org/).

![Class Diagram](class%20diagram.png)

Color code of classes:

* green: implemented

* yellow: still to do.

# Check if tests pass
For this project I used [pytest](https://docs.pytest.org/en/latest/) to write my tests. To try if every tests pass on your side, just type:

```bash
py.test tests/
```

If you want to improve mines feel free to help.
