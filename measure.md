# Measuring model performance

## Installation

The script in `prepare.sh` creates a virtualenv with Python 3.5, installs the
prerequisites in it and downloads and unpacks the ad-versarial dataset and the
model (if you have the .tgz files from
https://github.com/ftramer/ad-versarial/releases/download/0.1 already, put them
into `tmp/` directory here and they will be used instead of downloading).

After the installation the virtualenv will be in `venv` and it needs to be
activated (`source venv/bin/activate`) in order to run the model.

## Running the model in measurement mode

I have added an additional script for measuring model performance. It's located
in [page-based/measure.py](page-based/measure.py). Usage is quite similar to
`page-based/classify.py`, which is described in the
[README](page-based/README.md) that comes with it. There are two additional
parameters:

- `supp_threshold` - determined which detected boxes end up in the JSON output.
  This needs to be configurable because at the moment post-processing depends
  on this parameter and setting it to a different value than `conf_threshold`
  (detection threshold) might result in additional boxes being detected (so
  some boxes with p > 0.5 only appear in the output if `supp_threshold` is below
  0.5). I haven't figured why this is happening and once this is resolved this
  parameter can be set to some low value and mostly ignored. The default value
  for this is 0.1.
- `match_threshold` - defines IoU value at which a detected box is considered
  a match for a marked box. Setting this high will only recognize very precise
  matches whereas low values will make even a small overlap sufficient. The
  default is 0.4.

Functional differences between `classify.py` and `measure.py` are as follows:

- `measure.py` understands a directory with region markup in a CSV file.
- `measure.py` outputs statistics on how detected boxes match marked regions.
- `measure.py` produces a `summary.json` file in the output directory, which
  contains the information about which images were processed, where the output
  was stored, detected boxes, marked regions and how well it matched.

There's a convenience shell script: [measure.sh](measure.sh) that activates the
virtualenv and runs `measure.py`. It takes source directory as a parameter plus
any additional flags that will be passed to `measure.py`. The output always
goes to `output/`.
