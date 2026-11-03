from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def fixture_check():
    spec = importlib.util.spec_from_file_location('dose_analysis', HERE / 'analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(100)
    ds = np.repeat(np.arange(12), 48)
    x = rng.normal(size=len(ds))
    y = -.4 * x + rng.normal(size=12)[ds] + rng.normal(size=len(ds))
    frame = pd.DataFrame({'dataset': ds.astype(str), 'cell_type': np.tile(np.arange(48),12).astype(str),
        'deficit': x, 'effect': y, 'encoder': rng.choice(['a','b'],len(ds)),
        'seed': rng.choice([42,43,44],len(ds)), 'arm': rng.choice(['A','B'],len(ds))})
    result, _ = module.analyze_frame(frame, 'implementation_fixture_only', 20, 7)
    assert result['reference_converged'] and result['reference_max_coefficient_error'] < 1e-6
    (HERE / 'implementation-check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    fixture_check()
