# Time Series Classification in Financial Markets

Published: draft  
Medium: [Time Series Classification in Financial Markets](https://medium.com/@kyle-t-jones/time-series-classification-in-financial-markets-e850174e9675)

Random-forest direction classification on a synthetic financial-style time series (lag features, temporal train/test split). Companion code for the article (`article.md`).

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ts-classification-run
```

Outputs:

| Path | Contents |
|------|----------|
| `outputs/figures/` | Analysis dashboard and probability timeline (PNG) |
| `outputs/results.json` | Run metadata, accuracy, and relative figure paths |

## Project layout

```
config.yaml                 # data, model, and output settings
config.local.yaml.example
pyproject.toml / uv.lock
src/ts_classification/    # features, model, plots, CLI
outputs/figures/            # generated PNGs (gitignored except .gitkeep)
_drafts/                    # work-in-progress exports
scripts/                    # optional BERT notebook export
notebooks/                  # original Jupyter notebook
tests/
article.md
classification_ts_clean.py  # thin delegate to the package
```

## Configuration

Edit `config.yaml`:

- `data.seed`, `data.n_samples`, `data.train_ratio` — synthetic series and temporal split
- `model.n_estimators`, `model.max_depth` — Random Forest hyperparameters
- `output.show` — set `true` to open interactive plot windows locally

Machine-specific overrides: copy `config.local.yaml.example` to `config.local.yaml` (gitignored).

## Optional: BERT classification

The notebook export under `scripts/` uses Hugging Face Transformers. Install extras first:

```bash
uv sync --extra bert
uv run python scripts/bert_time_series_classification.py
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
