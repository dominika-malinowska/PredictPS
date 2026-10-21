# PredictPS

Pixel-wise prediction of persistent scatterer (PS) density from Sentinel-1 global coherence rasters and OpenStreetMap infrastructure maps.

## Installation

Requires Python 3.9.18 on Linux x86_64. The lockfile installs the PyTorch 2.0.1 CUDA 11.8 wheel (`torch==2.0.1+cu118`), not the CPU wheel from PyPI. macOS, native Windows, and CPU-only machines cannot use this lockfile as-is.

A CUDA GPU is required to train, test, or predict. `scripts/train.py`, `scripts/test_predict.py`, and `scripts/predict_region.py` build a Lightning trainer with `accelerator="gpu"`. Importing the package does not need a GPU or any rasters.

```bash
poetry install
```

Check the environment:

```bash
poetry run python -c "import torch, ps_density; print(torch.__version__, torch.cuda.is_available())"
```

`osmconvert` and the AWS CLI are external programs. They are used only by `scripts/predict_region.py`. Install `osmconvert` and put it on `PATH`, or set `osmconvert` in `config/paths.yaml` to the binary path. The AWS CLI is used to download a public coherence tile (`aws s3 cp --no-sign-request`); no AWS credentials are required.

## Configuration

File paths live in `config/paths.yaml`. The committed template is `config/paths.example.yaml`.

```bash
cp config/paths.example.yaml config/paths.yaml
# Edit data_root so it points at your raster directory
```

Override the root without editing the file:

```bash
export PREDICTPS_DATA_ROOT=/path/to/data
```

Use a different YAML file:

```bash
export PREDICTPS_PATHS_FILE=/path/to/my_paths.yaml
```

`config/paths.yaml` is gitignored. `data/` contents are gitignored except this repository's placeholder files. Do not commit rasters, checkpoints, or `.venv/`.

Torchgeo reads directories of `*.tif` files. A path such as `input/global_coh_NL/summer_vv_rho` must be a folder of GeoTIFFs, not a single file. The paper script (`scripts/data_analysis_for_the_first_paper.py`) is separate: it opens individual GeoTIFFs with PIL.

Train and test use different OSM folders (`input/OSM_input_updated` and `input/OSM_input_channel_v2`). Keep both keys unless you have confirmed the rasters are interchangeable.

## Required inputs

Set `data_root` before running a script. Nothing below is in git.

### Train (`scripts/train.py`)

- `{data_root}/input/global_coh_NL/summer_vv_rho/*.tif`
- `{data_root}/input/OSM_input_updated/*.tif`
- `{data_root}/input/mask/mask_EGMS/*.tif`
- a CUDA GPU

The Netherlands bounding box (longitude 3–8, latitude 50–54), patch size, and learning-rate settings stay in the script.

Checkpoints are written under `{data_root}/trainingnn_results/`. TensorBoard logs go in that experiment's `logs/` directory.

### Test on the Netherlands (`scripts/test_predict.py`)

Same coherence and mask folders as training, plus:

- `{data_root}/input/OSM_input_channel_v2/*.tif`
- `{data_root}/best_models_15122023/seed_54321_20231214-203338.ckpt` (loaded, then replaced)
- `{data_root}/trainingnn_results/pixel_wise_regression_model_training_updated_OSM/seed_54321_20240612-132349.ckpt` (the checkpoint the rest of the script uses)
- optional plot context: `{data_root}/input/Google_satellite/*.tif`

### Rome block in `scripts/test_predict.py`

- `{data_root}/input/Italy/coh_Rome_cropped/`
- `{data_root}/input/Italy/OSM_Rome_combined/`
- `{data_root}/input/Italy/SNT/msk/`
- `{data_root}/input/Italy/Google_satellite/`
- `{data_root}/best_models_1in_18122023/seed_54321_20231218-105437.ckpt` for the one-channel comparison

### Predict one 1×1 degree tile (`scripts/predict_region.py`)

- `{data_root}/best_models_15122023/seed_54321_20231214-203338.ckpt`
- an OSM `.pbf` already on disk. The default name is `slaskie-latest.osm.pbf`, placed in the experiment directory (`{data_root}/trainingnn_results_paper_revisions/pixel_wise_regression_run_predictions/`). Change `predict.osm_pbf_name` if your file has another name.
- `osmconvert` on `PATH` (or `osmconvert` in `paths.yaml`)
- the AWS CLI, for the public Sentinel-1 global coherence download
- a CUDA GPU

The example area of interest is Silesia (`predict.aoi_lon`, `predict.aoi_lat` in `paths.yaml`). The region must sit inside one 1×1 degree tile. Coherence season is summer in the northern hemisphere and winter in the southern hemisphere.

### Paper figures only

Not required to train or predict. `scripts/data_analysis_for_the_first_paper.py` reads these single GeoTIFFs under `data_root`:

- `input/global_coh_NL/{summer,spring,fall,winter}_vv_rho.tif`
- `input/mask/mask_EGMS/r_nb_pts_max_all.tif`
- `input/ndvi_gee/new/mean_ndvi_S2_wi_nan_filled_resized.tif`
- `input/ndvi_gee/new/mean_ndvi_S2_su_nan_filled_resized.tif`
- `input/ndvi_gee/new/std_ndvi_S2_wi_nan_filled_resized.tif`
- `input/ndvi_gee/new/std_ndvi_S2_su_nan_filled_resized.tif`
- `OSM_buildings/buildings.tif`, `railway.tif`, `highway.tif`, `highway_motorway.tif`, `highway_trunk.tif`, `highway_primary.tif`, `highway_secondary.tif`
- `input/OSM_input_channel_v2/OSM_combined_v2.tif`
- `merged_prediction_fixed_stiching_20240607_204020.tif`
- `water_land_mask.tif`
- `input/Italy/SNT/msk/Rome_nb_pts_des_cropped.tif`
- `merged_prediction_fixed_stiching_Rome_20240610_123734.tif`
- shapefiles under `EGMS_data_for_paper_plots/`

Plots are written to `{data_root}/plots_paper_final_version/`.

## Run

From the repository root, after `poetry install` and after `data_root` is set.

**Recreate the pipeline** (no interactive figures). These scripts run from top to bottom. They are not library modules.

```bash
poetry run python scripts/train.py
poetry run python scripts/test_predict.py
poetry run python scripts/predict_region.py
poetry run python scripts/data_analysis_for_the_first_paper.py
```

Paper figures come only from `scripts/data_analysis_for_the_first_paper.py`.

**Inspect rasters, sampling, or a few predictions, or change a seed or area of interest before a long run.** Open these notebooks in the Poetry environment (`poetry install --with dev` installs the Jupyter kernel):

- `notebooks/train.ipynb` — input stack, sampling map, optional training
- `notebooks/test_predict.ipynb` — a few test patches, optional Rome patch
- `notebooks/predict_region.ipynb` — infrastructure and coherence previews, then prediction

The notebooks call `ps_density`. They do not redefine the model or the raster readers. Skip the training or prediction cells if you only want the plots.

A data-free check:

```bash
poetry run python -c "import ps_density"
poetry run python -m py_compile scripts/train.py scripts/test_predict.py scripts/predict_region.py scripts/data_analysis_for_the_first_paper.py
```

## Repository structure

```text
config/paths.example.yaml          # path template (committed)
config/paths.yaml                  # your local paths (gitignored)
src/ps_density/                    # library: datasets, Lightning task, OSM prep, plots
scripts/train.py                   # train on the Netherlands stack
scripts/test_predict.py            # test, Netherlands maps, Rome transfer
scripts/predict_region.py          # predict one new 1x1 degree tile
scripts/data_analysis_for_the_first_paper.py
notebooks/train.ipynb              # optional plots and parameter edits for training
notebooks/test_predict.ipynb       # optional prediction-patch plots
notebooks/predict_region.ipynb     # optional regional-prediction plots
data/                              # optional local data root; rasters are not committed
```

`src/ps_density/coordinates_selection.py` is an interactive latitude/longitude helper. The scripts do not call it.

## License

MIT. See [LICENSE](LICENSE).
