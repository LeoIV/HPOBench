import numpy as np

from hpobench.benchmarks.nas.tabular_benchmarks import NavalPropulsionBenchmarkOriginal

if __name__ == '__main__':
    b = NavalPropulsionBenchmarkOriginal()
    config = b.get_configuration_space(seed=1).sample_configuration()
    print(config)
    # make all int64 values into int
    config = {k: v.item() if isinstance(v, np.int64) or isinstance(v, np.float64) else v for k, v in config.items()}

    print(b.get_configuration_space(1))
    result_dict = b.objective_function(configuration=config, rng=1)
    print(result_dict)

"""
need to install nas_benchmarks for this to work

git clone https://github.com/automl/nas_benchmarks.git
cd nas_benchmarks
python setup.py install
"""
