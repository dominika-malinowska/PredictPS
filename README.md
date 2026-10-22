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

`osmconvert` and the AWS CLI are external programs. Install `osmconvert` and put it on `PATH`, or set `predict.osmconvert` in `config/paths.yaml` to the binary path. The AWS CLI is used to download public coherence tiles (`aws s3 cp --no-sign-request`); no AWS credentials are required. They are used by `scripts/predict_region.py` and by the training-data prep scripts.

OpenStreetMap **`.osm.pbf`** extracts are **not** downloaded by the repository scripts. Download them yourself from [Geofabrik](https://download.geofabrik.de/) (for example the Netherlands extract for training, or a regional extract such as Silesia for the default predict example). The published training data used a Geofabrik Netherlands build from **26 October 2023**; you may reuse that snapshot or download a newer file — infrastructure rasters may differ slightly if OSM has changed.

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

Torchgeo accepts either one GeoTIFF or a directory of `*.tif` files. Coherence, OSM, and the EGMS mask in `paths.yaml` are the merged files (`summer_vv_rho.tif`, `OSM_combined.tif`, `r_nb_pts_max_all.tif`). Per-tile rasters under a sibling `tiles/` folder are not read. Rome and Google satellite paths stay directories. The paper script (`scripts/data_analysis_for_the_first_paper.py`) is separate: it opens individual GeoTIFFs with PIL.

`config/paths.yaml` is grouped by workflow (`prepare`, `train`, `test`, `predict`, `paper`). Fill only the block for the script you are running. Train and test share the same coherence, OSM, and EGMS mask files.

## Required inputs

Set `data_root` before running a script. Nothing below is in git.

### Prepare training inputs

The `prepare:` block sets `aoi_lon` / `aoi_lat` (same idea as predict), `prepare.osm_pbf`, and EGMS paths. 1×1 degree tiles are derived from that box. Optional `exclude_tiles` lists south-west corners to skip, or use `exclude_tiles: []` for the full rectangle. The coherence and OSM mosaics are written to `train.input.coherence_summer` and `train.input.osm`. Tile GeoTIFFs go in `tiles/` next to those files. The EGMS density raster is written by QGIS to `train.input.mask_egms`.

```bash
poetry run python scripts/prepare_data/prepare_coherence.py
poetry run python scripts/prepare_data/prepare_osm.py
```

`prepare_coherence.py` downloads `summer_vv_rho` tiles from the public Sentinel-1 global coherence bucket and mosaics them. Pass `--season`, `--polarisation`, or `--metric` to change the product.

`prepare_osm.py` clips `prepare.osm_pbf` per tile with `osmconvert`, rasterizes buildings/roads/rail with the same values as `scripts/predict_region.py`, and mosaics the tiles. `prepare.osm_pbf` may be one **`.osm.pbf`** or a directory of them. `osmconvert` accepts only one `.pbf` per call, so each extra extract is converted to `.o5m` once and the tile clip merges those files. A tile that crosses a country border is filled from all of them. Download the extracts from [Geofabrik](https://download.geofabrik.de/) yourself. The script does not fetch them.

The Zenodo deposit includes `OSM_combined.tif`, the infrastructure raster used for training. Per-tile GeoTIFFs are not included. It also includes three Geofabrik extracts from 26 October 2023 so the preparation script can be tried: `netherlands-latest.osm.pbf`, `drenthe-latest.osm.pbf`, and `nordrhein-westfalen-latest.osm.pbf`. Drenthe is already covered by the Netherlands extract. Belgium and the other German states used for the full raster are omitted because of file size. Download any further `.osm.pbf` files from [Geofabrik](https://download.geofabrik.de/). These extracts and the raster are derived from OpenStreetMap data © OpenStreetMap contributors, available under the Open Database License.

Open `scripts/prepare_data/prepare_egms.ipynb` in the Poetry environment. Automated cells filter the EGMS link list and merge split shapefiles. The EGMS website download and the QGIS density conversion stay manual (see the notebook). Training reads the three merged GeoTIFFs named in `train.input`; this notebook does not produce the mask raster.

### Train (`scripts/train.py`, `notebooks/train.ipynb`)

Only the `train:` block is required.

- `train.input.coherence_summer` — default `{data_root}/input/global_coh_NL/summer_vv_rho/summer_vv_rho.tif`
- `train.input.osm` — default `{data_root}/input/OSM_input_updated/OSM_combined.tif`
- `train.input.mask_egms` — default `{data_root}/input/mask/mask_EGMS/r_nb_pts_max_all.tif`
- a CUDA GPU

The Netherlands bounding box (longitude 3–8, latitude 50–54), patch size, and learning-rate settings stay in the script.

Checkpoints are written under `train.output.training_dir` / `train.output.experiment_name` (default `{data_root}/trainingnn_results/pixel_wise_regression_model_training_updated_OSM/`). TensorBoard logs go in that experiment's `logs/` directory.

### Test on the Netherlands (`scripts/test_predict.py`, `notebooks/test_predict.ipynb`)

The `test:` block uses the same coherence, OSM, and mask files as training, plus:
- `test.checkpoints.netherlands_20231214` — default `{data_root}/best_models_15122023/seed_54321_20231214-203338.ckpt` (loaded, then replaced)
- `test.checkpoints.netherlands_updated_osm` — default `{data_root}/trainingnn_results/pixel_wise_regression_model_training_updated_OSM/seed_54321_20240612-132349.ckpt` (the checkpoint the rest of the script uses)
- `test.checkpoints.one_channel_20231218` — default `{data_root}/best_models_1in_18122023/seed_54321_20231218-105437.ckpt` for the one-channel comparison
- optional plot context: `test.input.google_satellite_dir` — `{data_root}/input/Google_satellite/*.tif`

Outputs go under `test.output.revision_results_dir` / `test.output.experiment_name`. Review plots go under `test.output.plots_review_dir`.

### Rome block in `scripts/test_predict.py`

The nested `test.rome:` keys:

- `test.rome.coherence_dir` — `{data_root}/input/Italy/coh_Rome_cropped/`
- `test.rome.osm_dir` — `{data_root}/input/Italy/OSM_Rome_combined/`
- `test.rome.mask_dir` — `{data_root}/input/Italy/SNT/msk/`
- `test.rome.google_satellite_dir` — `{data_root}/input/Italy/Google_satellite/`
- `test.rome.plots_review_dir` — `{data_root}/plots_review_Rome/`

### Predict one 1×1 degree tile (`scripts/predict_region.py`, `notebooks/predict_region.ipynb`)

Only the `predict:` block.

- `predict.checkpoint` — default `{data_root}/best_models_15122023/seed_54321_20231214-203338.ckpt`
- an OSM **`.pbf`** from [Geofabrik](https://download.geofabrik.de/) that covers your AOI (for the default Silesia example, a regional extract such as `slaskie-latest.osm.pbf`). Place it in the experiment directory (`predict.output.revision_results_dir` / `predict.output.experiment_name`) with the name given by `predict.osm_pbf_name`. The script does not download it; it only runs `osmconvert` and rasterization.
- `osmconvert` on `PATH` (or `predict.osmconvert` in `paths.yaml`)
- the AWS CLI, for the public Sentinel-1 global coherence download
- a CUDA GPU

The example area of interest is Silesia (`predict.aoi_lon`, `predict.aoi_lat`). The region must sit inside one 1×1 degree tile. Coherence season is summer in the northern hemisphere and winter in the southern hemisphere.

### Paper figures only

The `paper:` block is not required to train or predict. `scripts/data_analysis_for_the_first_paper.py` writes to `paper.plots_dir` and reads these single GeoTIFFs under `data_root`:

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

Plots are written to `paper.plots_dir` (default `{data_root}/plots_paper_final_version/`).

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
poetry run python -m py_compile scripts/train.py scripts/test_predict.py scripts/predict_region.py scripts/data_analysis_for_the_first_paper.py scripts/prepare_data/prepare_coherence.py scripts/prepare_data/prepare_osm.py
```

## Repository structure

```text
config/paths.example.yaml          # path template (committed)
config/paths.yaml                  # your local paths (gitignored)
src/ps_density/                    # library: datasets, Lightning task, OSM prep, plots, path loader
scripts/train.py                   # train on the Netherlands stack
scripts/test_predict.py            # test, Netherlands maps, Rome transfer
scripts/predict_region.py          # predict one new 1x1 degree tile
scripts/data_analysis_for_the_first_paper.py
scripts/prepare_data/prepare_coherence.py
scripts/prepare_data/prepare_osm.py
scripts/prepare_data/prepare_egms.ipynb
notebooks/train.ipynb              # optional plots and parameter edits for training
notebooks/test_predict.ipynb       # optional prediction-patch plots
notebooks/predict_region.ipynb     # optional regional-prediction plots
data/                              # optional local data root; rasters are not committed
```

`src/ps_density/coordinates_selection.py` is an interactive latitude/longitude helper. The scripts do not call it.

## License

MIT. See [LICENSE](LICENSE).
