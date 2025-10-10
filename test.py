from hpobench.benchmarks.ml import XGBoostBenchmark
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark

if __name__ == '__main__':

    b = XGBoostBenchmark(task_id=10101)
    config = b.get_configuration_space(seed=1).sample_configuration()
    print(config)
    print(b.get_configuration_space(1))
    result_dict = b.objective_function(configuration=config,
                                       fidelity={"n_estimators": 128}, rng=1)
    print(result_dict)
