# MCI connectivity forecasting

Code for *Windowed Connectivity Variability for Two-Year Clinical Severity
Prediction in Mild Cognitive Impairment* by Yury Maximov, Aleksandr Volkov,
Tamoghna Chattopadhyay, Paul M. Thompson, and Ildar Gabitov.

The paper is under review for the **2027 IEEE International Symposium on
Biomedical Imaging (ISBI 2027)**.

We compare clinical covariates alone with models that add static functional
connectivity, windowed connectivity dispersion, or both. The repository includes
the prediction code and a reproducible synthetic example. Saved study results,
row split plans, and saved synthetic outputs live in the private
[yury-rd/mci-forecast-data](https://github.com/yury-rd/mci-forecast-data) companion.

## Run the example

Use Python 3.11 or newer. The reference environment uses Python 3.14 and the
versions pinned in `requirements.txt`.

```sh
git clone https://github.com/yury-rd/mci-forecast.git
cd mci-forecast
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --output results/demo.json
```

On Windows, activate the environment with `.venv\Scripts\activate`.
The example fits 48 synthetic rows and reports mean absolute error (MAE) for
each model and repetition, along with the selected settings. Its errors differ
from the clinical results. Numerical thread pools use one thread.

## Private study data

The public checkout and its tests do not require study data. Authorized users
can initialize the private companion mounted at `data/`:

```sh
git submodule update --init data
python run.py --mode summary
```

GitHub authentication and permission to access `yury-rd/mci-forecast-data` are
required. A missing checkout produces an actionable error. `data/results.json`
contains the saved aggregate results, `data/splits.json` contains the study's
row split plan, and `data/expected_demo.json` preserves the saved synthetic
output. These artifacts are absent from public source and distributions.

The demo generates a new deterministic synthetic split plan in memory. It
therefore does not replay the saved study splits or historical demo output.

## Fit your own data

Prepare an NPZ file with one row per participant:

| Array | Shape | Contents |
| --- | --- | --- |
| `target` | `(N,)` | Future clinical severity |
| `static` | `(N, 21)` | Full-record network correlations |
| `dispersion` | `(N, 21)` | Standard deviations of window correlations |
| `clinical_numeric` | `(N, C)` | Numeric clinical and technical covariates |
| `categories` | `(N, K)` | Unicode categorical values; use `(N, 0)` if absent |
| `participant_ids` | `(N,)` | Optional unique identifiers |

Numeric arrays must be finite. NPZ loading disables pickle. The combined MRI
block is constructed from `static` and `dispersion`.

Supply a JSON split plan with `seed`, `fold`, `train`, `test`, and `inner` for
each outer fold. Each inner fold has its own `train` and `test` lists. Indices
refer to rows in the complete input array, including within inner folds.
Training and test sets must be disjoint and cover their parent fold. Test folds
must cover each participant exactly once per repetition.

```sh
python run.py --input local_data/features.npz --splits local_data/splits.json --output results/local.json
```

The private `data/splits.json` contains the original study row plan. The public
`forecast.synthetic_splits()` generates five outer folds repeated for three seeds,
with three inner folds, for the synthetic example. Reproducing the clinical analysis requires authorized ADNI data and
the matching participant-order manifest. Obtain access through
[ADNI](https://adni.loni.usc.edu/); participant records and derived individual
features are not distributed here.

## Methods and files

`features.py` extracts static connectivity and dispersion from a prepared
samples-by-seven network time series after preprocessing and M24 nuisance
regression. `extract(timeseries, tr_seconds)` uses all 21 upper-triangle Pearson
correlations, a window width of `max(8, round(60/TR))`, and stride
`max(1, round(30/TR))`. Dispersion is the sample standard deviation (`ddof=1`);
correlations are not Fisher-transformed. The module also provides common-phase
surrogates and window selections for the temporal-coverage controls. Image
preprocessing and construction of the clinical target precede this interface.

`forecast.py` fits scalers, categorical vocabularies, PLS projections, and ridge
models on training rows only. Inner folds select 2, 5, or 10 PLS components and
a ridge penalty of 0.1, 1, 10, or 100. The clinical-only model selects its penalty
on the same folds. MRI columns are standardized and divided by the square root
of block width; the combined representation is one 42-coordinate block. PLS
uses training outcomes with its additional scaling disabled. Training matrices
must meet the PLS rank requirement. `run.py` runs these comparisons or reads the
private saved results. Companion data paths are resolved relative to the script; paths
supplied on the command line are relative to your working directory.

## Check the code

```sh
python -m unittest discover -s tests -v
```

Tests cover feature extraction, spectral invariants, training/test isolation,
split validation, deterministic synthetic predictions, private-data error
handling, and execution from another working directory. The workflow template under `.github/` runs the public tests
and generated demo without private credentials. To enable it, copy the template
into `.github/workflows/` using credentials with GitHub workflow permissions.

## Citation and license

Please cite the paper as a manuscript under review. The author list, title, and
conference are recorded in [CITATION.cff](CITATION.cff). The repository is
[yury-rd/mci-forecast](https://github.com/yury-rd/mci-forecast).

The public code and synthetic generator use the [MIT license](LICENSE). Data
licenses and access rules travel with the private companion. ADNI data remain
subject to their data use agreement.
