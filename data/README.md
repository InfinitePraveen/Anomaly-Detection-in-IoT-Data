# Data

This project uses the **Numenta Anomaly Benchmark (NAB)** as an open-source time-series benchmark.

The notebooks download these files automatically:

- `art_daily_small_noise.csv` — normal sensor behavior used for training.
- `art_daily_jumpsup.csv` — sensor behavior containing anomalous periods used for testing.

Source:
https://github.com/numenta/NAB

Raw files used by the notebooks:

- https://raw.githubusercontent.com/numenta/NAB/master/data/artificialNoAnomaly/art_daily_small_noise.csv
- https://raw.githubusercontent.com/numenta/NAB/master/data/artificialWithAnomaly/art_daily_jumpsup.csv

The CSV files are intentionally not committed here because they are downloaded reproducibly by the notebooks.
