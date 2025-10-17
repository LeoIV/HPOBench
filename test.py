import tensorflow as tf


from hpobench.benchmarks.ml import XGBoostBenchmark, TabularBenchmark
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from hpobench.benchmarks.nas.tabular_benchmarks import NavalPropulsionBenchmarkOriginal

if __name__ == '__main__':




    b = NavalPropulsionBenchmarkOriginal()
    config = b.get_configuration_space(seed=1).sample_configuration()
    print(config)
    print(b.get_configuration_space(1))
    result_dict = b.objective_function(configuration=config, rng=1)
    print(result_dict)

"""
need to install nas_benchmarks for this to work

git clone https://github.com/automl/nas_benchmarks.git
cd nas_benchmarks
python setup.py install
"""