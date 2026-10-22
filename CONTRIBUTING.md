# Contributing

Issues and pull requests are welcome. Before making a change, describe the
problem or proposed improvement and keep pull requests focused.

## Development checks

1. Use Python 3.10 or newer and install dependencies with
   `pip install -r requirements.txt`.
2. Run `pytest` and `python scripts/smoke_test.py --epochs 3 --use-synthetic`.
3. For code changes, run `ruff check .` when Ruff is installed.
4. Do not commit MELD data, generated outputs, model checkpoints, credentials,
   or other private artifacts.

Please document changes to configuration and commands, add tests for changed
behavior, and label synthetic results as synthetic. Real MELD performance
claims must include the split, configuration, and reproducible run details.
