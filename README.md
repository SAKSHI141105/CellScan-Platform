# CellScan

A two-pipeline breast tissue classification system — one path over the Wisconsin
Diagnostic Breast Cancer (WDBC) tabular dataset, one path over histopathology
image patches — with supervised and unsupervised models on both sides, SHAP/LIME/
Grad-CAM explanations, and a FastAPI + React frontend that ties it together.

This is a research/portfolio project. It is **not a diagnostic tool** — see
[Ethical disclaimer](#ethical-disclaimer) below and the About page in the dashboard.

## Why two pipelines, and why unsupervised too

Most breast-cancer-ML writeups pick one dataset and one model family. The point
here was to build something closer to what an actual hospital data team deals
with: structured records *and* imagery, and — critically — the assumption that
most of that data won't come pre-labeled. Clustering and the autoencoder
anomaly detector are there to answer "does the feature space separate the two
classes on its own, without ever seeing a diagnosis label?" That's the
realistic starting point when you're handed a pile of unlabelled historical
scans and asked to find something useful in them.

## Results (tabular pipeline, current run)

Held-out test set, 569-sample WDBC dataset, 20% held out:

| model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| ensemble_voting | 0.983 | 1.000 | 0.952 | 0.976 | 0.991 |
| ensemble_stacking | 0.983 | 1.000 | 0.952 | 0.976 | 0.991 |
| logistic_regression | 0.974 | 0.976 | 0.952 | 0.964 | 0.997 |
| mlp | 0.974 | 0.976 | 0.952 | 0.964 | 0.991 |
| random_forest | 0.965 | 0.975 | 0.929 | 0.951 | 0.995 |
| xgboost | 0.965 | 0.975 | 0.929 | 0.951 | 0.991 |
| svm | 0.965 | 0.975 | 0.929 | 0.951 | 0.991 |

Sorted by recall on purpose — a missed malignant case costs a lot more than a
false alarm, so that's the number that decides the ranking, not accuracy.

SMOTE vs. baseline (logistic regression, isolating the resampling effect from
model choice): on this dataset SMOTE didn't move recall or F1 at all (0.929 /
0.951 both ways) — WDBC's 63/37 split just isn't imbalanced enough for
resampling to matter much. It's included anyway because on a genuinely skewed
dataset (or the IDC image set, which runs closer to 70/30 malignant-minority
in some slides) the effect is usually larger, and the comparison is worth
having in the pipeline either way.

Clustering vs. true diagnosis (no labels used during clustering):

| method | silhouette | ARI | NMI |
|---|---|---|---|
| kmeans | 0.324 | 0.486 | 0.371 |
| hierarchical | 0.305 | 0.578 | 0.455 |
| dbscan | 0.272 | -0.023 | 0.079 |

KMeans and hierarchical clustering recover a meaningful chunk of the true
diagnosis structure (ARI 0.49–0.58) without ever seeing the label — reasonable
evidence the 30 features are genuinely discriminative on their own. DBSCAN,
with the default `eps=1.5`, mostly returns noise on this feature space (see
[Design decisions](#design-decisions--trade-offs)) — that's a real finding
about density-based clustering on this data, not a bug we papered over.

Image-pipeline numbers aren't included here because the image dataset isn't
vendored in this repo (see [Image dataset setup](#image-dataset-setup)) —
run `scripts/train_image.py` after downloading it and the same tables will
populate from that run.

## Setup

Python side:

```bash
python -m venv .venv
source .venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate cellscan
```

Frontend side (Node 20+):

```bash
cd frontend
npm install
```

No dataset download is required for the tabular pipeline — it falls back to
the copy of WDBC bundled in scikit-learn if `data/raw/wdbc.csv` isn't present.
The image pipeline does need a download; see below.

## Running things

```bash
# EDA figures -> reports/figures/
python -m scripts.eda_tabular

# full tabular run: cleaning, feature selection, SMOTE comparison,
# clustering, 5-model tuning + ensembles, saves everything to data/models/tabular/
python -m scripts.train_tabular

# image pipeline (needs the dataset — see below)
python -m scripts.train_image

# don't want to wait on a dataset download to try the Upload Image or
# Mammography pages? generates ImageNet-only checkpoints for both (no
# histopathology/mammography training at all) so the upload -> Grad-CAM ->
# report flow works end to end. Clearly labeled "DEMO" everywhere in the
# UI/exports — see Design decisions below for why. --target histopathology
# or --target mammography to build just one.
python -m scripts.generate_demo_weights
# (same thing, more memorable name: python -m scripts.setup_demo_environment)

# unit tests
pytest
```

The app is two processes: a FastAPI backend serving the ML pipelines, and a
Vite/React frontend. Easiest path — one command starts both, waits until
they're actually answering, and opens the browser for you:

```bash
python run.py
```

(`--no-browser` if you don't want the tab auto-opened.) This exists because
"run these two things in two terminals" is exactly the instruction people
skip half of, then report the app as broken when it's really just the API
process that never started — `run.py` makes that failure mode structurally
impossible: it health-checks both services before declaring success, and if
either one dies it shuts the other down rather than leaving you with a
frontend silently unable to reach its API.

Prefer to run them yourself (e.g. for `--reload` during backend development):

```bash
# terminal 1 — API on :8000
uvicorn src.api.main:app --reload --port 8000

# terminal 2 — frontend on :5173 (proxies /api/* to :8000, see frontend/vite.config.ts)
cd frontend
npm run dev
```

Either way, open `http://localhost:5173` — not :8000, that's the bare API.
The dashboard opens directly on the Overview page, no sign-in step.

The Clinical Data page works immediately
even before you've run `train_tabular.py` — the API falls back to a quick
untuned RandomForest trained in-memory so the app is never a dead end on a
fresh clone. Run the real training script for the tuned ensemble + SHAP
explanations backed by the actual saved model. The Model Performance and
Cluster Explorer pages read from `reports/figures/` and the saved models
respectively, so they populate once `train_tabular.py` has run.

### Image dataset setup

The image pipeline targets BreakHis or the Kaggle Breast Histopathology
Images (IDC) dataset — neither is included here (several GB, redistribution
terms vary). Download one, then arrange it as:

```
data/raw/histopathology/
    0/    (or benign/)     *.png
    1/    (or malignant/)  *.png
```

`config/config.yaml` → `paths.image_root` points here by default. Both the
`{0,1}` and `{benign,malignant}` folder-name conventions are supported since
BreakHis and the Kaggle IDC set don't agree with each other.

## Folder structure

```
run.py                       single-command launcher — starts the API + frontend
                              together and health-checks both (see Running things)
config/config.yaml          all hyperparameters, paths, model grids — nothing
                             hardcoded in the pipeline code
src/
  data_preprocessing/        loading, cleaning, SMOTE, image preprocessing
  feature_engineering/       correlation-based feature selection, GLCM/edge features
  models/                    clustering, autoencoders, the 5-model tabular zoo,
                              custom CNN + transfer learning
  models/mammography/         PyTorch lesion-guided ResNet50 — a same-architecture
                              port of research/mammography_generalization's model
                              (see Research module below for why it's a port, not
                              an import)
  explainability/            SHAP + LIME (tabular), Grad-CAM (image + mammography)
  services/                  framework-agnostic prediction + explanation logic
                              (tabular_service, image_service, mammography_service,
                              report_service) — the layer the API calls into
  api/                       FastAPI app — main.py + routers/ (tabular, image,
                              mammography, reports, clusters)
  utils/                     config loader, logging, shared metrics
frontend/                    React + TypeScript + Vite + Tailwind UI
  src/pages/                 one file per route (clinical data, upload image,
                              mammography, model performance, cluster explorer, about)
  src/components/ui/         hand-rolled component primitives (button, card,
                              tabs, table, dropzone, animated number, spotlight card)
  src/contexts/theme-context.tsx   light/dark/system theme provider
  src/lib/api.ts             typed fetch wrappers around the FastAPI routes
scripts/                     entry points — eda_tabular, train_tabular, train_image
tests/                       unit tests for the preprocessing functions
reports/figures/             generated EDA plots + metrics CSVs (gitignored contents,
                              folder tracked)
data/                        raw/processed data + trained model artifacts (gitignored)
research/
  mammography_generalization/  separate PyTorch research pipeline — see below
```

## Research module: mammography cross-dataset generalization

`research/mammography_generalization/` is a second, self-contained pipeline
in this same repo — a PyTorch project testing whether an auxiliary
lesion-segmentation loss makes a mammography classifier's backbone
generalize better across scanners/institutions (CBIS-DDSM → zero-shot on
VinDr-Mammo/INbreast), versus a classification-only baseline trained the
same way otherwise.

It's kept structurally separate as its own venv/tests/training pipeline —
different question (cross-dataset generalization, not single-dataset
classification), different training/eval workflow (`python train.py
--config ...` from the terminal, not something a web request drives) — but
the **model architecture is now also served live** from the dashboard's
Mammography page, backed by `src/models/mammography/` and
`src/services/mammography_service.py` in the main app.

That inference surface is a same-architecture *port* of the research
pipeline's model, not a Python import of it — both this app and the
research pipeline define a top-level `src` package, so importing one into
the other's process would collide. `src/models/mammography/backbone.py` and
`lesion_guided_model.py` are structurally identical copies of their
`research/mammography_generalization/src/models/` counterparts (same
layers, same names), which means a checkpoint produced by
`research/mammography_generalization/train.py` loads directly into the
dashboard's inference path — you train there, you serve here, without a
conversion step. If you change the architecture in one place, change it in
the other, or checkpoints stop being interchangeable (there's no test
enforcing this yet — worth adding if the architecture starts moving).

`python -m scripts.generate_demo_weights --target mammography` gets the
Mammography page working immediately with ImageNet-only weights (see
[Design decisions](#design-decisions--trade-offs) for how that's labeled).
For a real checkpoint, train via the research pipeline —
[research/mammography_generalization/README.md](research/mammography_generalization/README.md)
covers its own setup, architecture, and the CBIS-DDSM/INbreast/VinDr-Mammo
data-access instructions (none of them are simple downloads) — then point
`config/config.yaml`'s `mammography.checkpoint_candidates` at the resulting
`.pt` file (or drop it at `data/models/mammography/lesion_guided_resnet50.pt`,
the default first candidate).

## Design decisions / trade-offs

**Column naming, sklearn vs. Kaggle.** `sklearn.datasets.load_breast_cancer`
names its columns `"mean radius"` / `"worst concave points"` / `"radius
error"`; the Kaggle WDBC CSV people usually reach for uses `radius_mean` /
`concave_points_worst` / `radius_se`. Rather than picking one convention and
making the other loader match it downstream, `tabular_preprocessing.py`
normalizes sklearn's names to the Kaggle convention at load time — every other
module (feature selection, the clinical-data form, the EDA script) only ever
sees one naming scheme.

**Recall as the tuning objective, not accuracy or F1.** All five
GridSearch/RandomizedSearch calls use `scoring="recall"`. On a class-balanced
toy metric like accuracy, a model that leans slightly conservative can look
identical to one that's actually catching more malignant cases. In a real
triage context the asymmetry between a false negative and a false positive is
large enough that it should show up in what the search optimizes for, not
just in how results get reported afterward.

**Correlation-pruning before importance-ranking, not instead of it.**
`feature_selection.py` drops one column from every >0.95-correlated pair
first (radius/perimeter/area mean are ~0.99 correlated with each other by
construction), *then* ranks what's left by Random Forest importance. Ranking
first would just surface three near-duplicate radius/perimeter/area features
in the top five and call it done.

**GridSearchCV for the small grids, RandomizedSearchCV for RF/XGBoost.** The
Random Forest and XGBoost grids are wide enough (3–4 values across
3 hyperparameters each) that a full grid search is needlessly slow for the
marginal gain over 25 random draws. Logistic Regression / SVM / MLP grids are
small enough that grid search is basically free — no reason to introduce
sampling noise there.

**DBSCAN's default `eps` isn't tuned for this feature space, on purpose.**
It mostly labels points as noise (see the clustering table above). Tuning
`eps` per-dataset would hide a genuinely useful observation: density-based
clustering doesn't just work out of the box on standardized 30-D biomedical
features the way it might on 2-D geospatial data. The config exposes
`clustering.dbscan_eps` if you want to sweep it yourself.

**Autoencoder trained on benign-only data.** Both autoencoders (dense for
tabular, conv for image) train exclusively on the benign class, then flag
high reconstruction error on unseen samples as a malignancy signal. Training
on the full mixed set would just teach the network to reconstruct everything
equally well, and the anomaly signal disappears — this only works because we
deliberately withhold the malignant class during training.

**The API's fallback tabular model.** `src/services/tabular_service.py`
trains an in-memory RandomForest if `data/models/tabular/ensemble_voting.joblib`
doesn't exist yet. This is flagged with a TODO in the code — it's a
reasonable call for making the app never be a dead end on a fresh clone, but
it's exactly the kind of implicit fallback you'd strip out before anything
resembling a real deployment, where "which model produced this number" needs
to be an unambiguous, auditable fact.

**fpdf2 over a browser-rendered PDF library.** The downloadable report uses
fpdf2 instead of, say, WeasyPrint or a headless-Chrome HTML-to-PDF route.
fpdf2 is pure Python with no system dependency (no wkhtmltopdf binary, no
Chromium), which matters more for "this has to work with one pip install on
someone else's machine" than the layout flexibility we're giving up.

**A real React frontend instead of Streamlit.** This started as a Streamlit
app, which is genuinely the right call for a fast internal ML demo — but it
caps out on UI quality (you're styling Streamlit's own DOM with injected CSS,
which only goes so far, and there's no way to run real component libraries
inside it). Once polished UI with proper light/dark/system theming became a
requirement, the honest fix was separating concerns properly: FastAPI serves
the ML pipelines as a JSON API (`src/api/`), and a Vite/React/TypeScript
frontend (`frontend/`) consumes it — the same architecture split you'd expect
on an actual product team, and the only way to get real animated components
and theme control instead of a styled-Streamlit ceiling. The ML/explainability
code in `src/` didn't need to change for this — only the code that used to
import `streamlit` directly moved into a framework-agnostic `src/services/`
layer that both the API and the training scripts can call.

**Two Python processes' worth of numpy had to agree.** Partway through
building the API, `joblib.load()` on the saved tabular models started
throwing `ValueError: <class 'numpy.random._mt19937.MT19937'> is not a known
BitGenerator`. Cause: the models had been pickled under whatever numpy got
resolved on an earlier `pip install` (2.x), and installing TensorFlow
afterward silently downgraded numpy to `<2.0` to satisfy its own pin —
breaking numpy's internal RandomState pickle format compatibility between the
two major versions. Fix was just re-running `train_tabular.py` against the
now-stable, `requirements.txt`-pinned environment; worth calling out because
it's a real failure mode of installing ML dependencies incrementally rather
than all at once from a lockfile.

**UMAP gets warmed at API startup, not computed lazily.** Measured locally:
the first `UMAP().fit_transform()` call in a fresh process takes ~90 seconds
— almost entirely numba JIT-compiling UMAP's internals, not the actual
projection math — while every call after that is near-instant for the rest
of the process's life. Making someone's first click on the Cluster Explorer's
UMAP tab eat that cost would look exactly like a hung server. `src/api/main.py`
fires a background thread on startup (`clusters.warm_projection_cache()`)
that computes and caches the 2D and 3D UMAP projections before anyone's
likely to have asked for them; the frontend also has a "this may take a
minute" notice as a fallback for the case where someone's fast enough to hit
it mid-warmup anyway.

**Plotly for 3D, recharts for 2D — not one library for both.** The Cluster
Explorer's 2D scatter plots stayed on recharts (already in use for the ROC
curves), but 3D needs actual WebGL, which is what pulled in Plotly. Rather
than replace recharts everywhere for consistency, `Scatter3D` is the one
component built on it, and it's lazy-loaded (`React.lazy` + `Suspense`) —
plotly.js alone is several MB, and there's no reason to make every visitor
download it just to view the 2D charts on every other page.

**The image-model demo fallback is loud on purpose.** `load_image_model()`
in `src/services/image_service.py` resolves, in order: a real trained
checkpoint, saved demo weights (`scripts/generate_demo_weights.py`), or —
if neither exists — builds an ImageNet-only model in memory on the spot. Every
tier past the first one sets `is_demo: true` on the API response, and the
frontend threads that flag through every surface that touches it: the
pre-upload status banner, the results table (a `DEMO` badge per row), the
specimen header, the risk gauge card, the clinical-summary text, and the
exported CSV/PDF (a bold warning line, not a footnote). This is more
plumbing than a quiet `if not model: use_random()` fallback would need, but
a fabricated prediction that *looks* like a real one is a genuinely bad
failure mode for anything in a medical-app shell — cheap to over-communicate
here, expensive to under-communicate.

While wiring this up, generating a demo checkpoint and reloading it in a
fresh process (which is the real deployment path — `train_image.py` saves,
`image_service.py` loads later, possibly after a restart) surfaced a bug
that would have hit the *real* trained-model path too: `build_transfer_model`
used an unregistered custom Keras layer and a `Lambda` closure, both of
which fail to deserialize outside the process that created them
("Unknown layer: 'ReplicateChannels'"). Fixed by registering the custom
layer with `@register_keras_serializable` and replacing the Lambda closure
with a proper layer that looks up its preprocessing function by name
instead of capturing it — see `src/models/image_cnn.py`. Worth calling out
because it's the kind of bug that's invisible until someone actually reloads
a saved model in a new process, which nothing in this project had done
before generating demo weights forced the issue.

**Mammography's "clinical indicators" are derived, not fabricated.** The
`explanation` block returned alongside a mammography prediction
(`estimated_lesion_area_fraction`, `attention_concentration`) isn't invented
copy — the first is the fraction of the model's own segmentation-decoder
output above threshold, the second is what share of the Grad-CAM heatmap's
total energy sits in its hottest 10% of pixels (a real measure of "is
attention localized on one structure or smeared across the image"). Both
are computed straight from the model's actual outputs for that image. They're
still not clinical measurements in any validated sense — with the demo
checkpoint specifically, an untrained decoder head means
`estimated_lesion_area_fraction` reliably comes back near 100% (its sigmoid
output has no reason to look like a real mask yet) rather than something
that looks plausible, which is itself a reasonably honest signal that
you're looking at a demo prediction if the labeling elsewhere weren't
already unmissable.

**No login gate, by design.** An earlier version of the dashboard had a mock
sign-in step (any name, any 6-digit code, no real backend session) —
removed since a fake-but-convincing-looking auth screen is a worse look for
anything adjacent to a medical tool than no login screen at all, and it
never gated anything real to begin with. The app now opens straight to the
Overview page.

**The GLCM texture chart is real per-image math, not a canned example.**
Both the histopathology and mammography prediction responses now include
`texture_features` — the same `extract_classical_features()` used in the
tabular clustering pipeline (`src/feature_engineering/texture_features.py`),
run against the actual preprocessed pixels of whatever was just uploaded.
Feeding it pure random noise during testing produced GLCM contrast in the
thousands (expected — noise has no spatial coherence at all), which is a
decent sanity check that it's really reading the image and not returning a
fixed example.

## Testing

`pytest` covers the preprocessing layer — column normalization, missing-value
imputation, stratified splitting, scaling, SMOTE balancing, and the image
preprocessing chain (grayscale/resize/CLAHE/denoise/normalize) plus GLCM/edge
feature extraction. It deliberately doesn't try to unit-test model training
itself (tuning + CV is slow and non-deterministic enough that it belongs in
the training scripts' own logged output, not a fast test suite).

## Ethical disclaimer

CellScan is a research and educational project. It is **not** a medical
device, has not been validated on an independent clinical cohort, and carries
no regulatory clearance (FDA, CE, or otherwise). Nothing in this repository
or its dashboard should influence an actual diagnostic or treatment decision.
A real clinical deployment of anything like this would require prospective
clinical validation and the corresponding regulatory approval process —
neither of which this project attempts.

## Limitations and future scope

- **Multi-modal fusion.** The two pipelines run independently; a patient with
  both a tabular record and an imaging study currently gets two separate
  scores rather than one fused prediction. Combining them — even a simple
  weighted average of calibrated probabilities — is the most obvious next
  step.
- **Federated learning.** Hospitals can't pool raw patient imagery across
  institutions for privacy/regulatory reasons. Training the image CNN in a
  federated setup (local gradients only, no raw data leaving the source) is
  the realistic path to a model trained on more than one institution's data.
- **Image pipeline validation.** Numbers for the image side aren't in this
  README because the dataset isn't vendored — the code path is complete and
  has been reviewed, but hasn't been exercised end-to-end against BreakHis or
  the full IDC set in this environment.
- **Calibration.** Predicted probabilities aren't currently calibrated
  (e.g. via Platt scaling / isotonic regression) — the SVM and ensemble
  outputs in particular shouldn't be read as true probabilities without that
  step.
