# WHIRLD

The main implementation is organized as follows:

- `models/simht.py`: deterministic SimHT predictor and HTBlock.
- `whirld.py`: wavelet residual dynamical diffusion, DWUNet, and recurrent wavelet context.
- `datasets/`: dataset loaders and evaluation visualizations.
- `run.py`: training and evaluation entry point.

## Environment

```bash
conda env create -f environment.yml
conda activate whirld
```

## Data

The experiments use SEVIR, MeteoNet, Shanghai Radar, and CIKM Radar. Download each dataset from its official or public source and provide its local path with `--data_path`.

- [SEVIR](https://nbviewer.org/github/MIT-AI-Accelerator/eie-sevir/blob/master/examples/SEVIR_Tutorial.ipynb)
- [MeteoNet](https://meteofrance.github.io/meteonet/english/data/rain-radar/)
- [Shanghai Radar](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2GKMQJ)
- [CIKM Radar](https://tianchi.aliyun.com/dataset/1085)

Alternatively, set `WHIRLD_DATA_ROOT` and use the default names:

```text
data/
|-- cikm.h5
|-- meteo_radar.h5
|-- shanghai.h5
`-- sevir/
```

## Training

The following example trains WHIRLD on MeteoNet:

```bash
python run.py \
  --backbone simht \
  --dataset meteo \
  --data_path /path/to/meteo_radar.h5
```



## Evaluation

Evaluate a saved checkpoint with:

```bash
python run.py \
  --eval \
  --backbone simht \
  --dataset meteo \
  --data_path /path/to/meteo_radar.h5 \
  --ckpt_milestone /path/to/checkpoint.pt
```
