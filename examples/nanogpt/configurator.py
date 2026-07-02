"""Poor Man's Configurator — exec()'d by train.py to handle CLI overrides.

Usage (from train.py):
    python train.py config/train_gpt2.py --batch_size=32 --wandb_log=False

Positional args (no '--') are treated as config file paths and exec()'d.
'--key=value' args override individual globals with type-safe literal_eval.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if "=" not in arg:
        # Config file path
        assert not arg.startswith("--")
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())  # noqa: S102
    else:
        # --key=value override
        assert arg.startswith("--")
        key, val = arg.split("=")
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])  # noqa: E721
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
