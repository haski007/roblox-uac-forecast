# Roblox Up-and-Coming top-10 forecaster

Entry for the make.fun bounty `roblox-up-and-coming-forecast-01` ("Predict Roblox's Up-and-Coming top 10").

`main.py` (Python 3.13, standard library only, deterministic) reads `/input/history.ndjson` and `/input/task.json`
and writes `/output/forecast.json`: for each of the top 10 places, a probability for every game on the last chart,
plus a best-guess top 10.

## Method

- From every pair of past reads of the chart it learns where a game at place r ends up after a time lag, pooled by
  place bin and lag, and smoothed over neighbouring places (wider further down the chart).
- A game's recent rank trend (momentum) shifts its starting place, for training pairs and for the forecast alike.
- One forecast serves all 72 hourly reads, so the lag-specific kernels are averaged over the window's horizons.
- Every game on the last chart is named at every place; at most 90% of a place is handed out, and the rest is the
  share of new entrants (new games take 15 to 45% of top-10 places by 3 days out).

## Run

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT, see `LICENSE`.
