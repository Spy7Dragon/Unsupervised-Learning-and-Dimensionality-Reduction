from sklearn import *

import comparison_generation
import io_manager
import pandas as pd
import traceback
import scipy


def perform_ica(data):
    new_training_features = None
    new_test_features = None
    try:
        model = decomposition.FastICA()
        best_name = comparison_generation.create_attribute_storage_name(data, model)
        best = io_manager.load_best(best_name)
        comparison_generation.update_model(model, best)
        if not io_manager.is_converged(best, best_name):
            best['n_components'] = comparison_generation.compare_mixing_numbers(model, data,
                                                                                'n_components',
                                                                                intervals=5,
                                                                                interval_size=1,
                                                                                start_index=2)
            comparison_generation.update_model(model, best)
            values = ['parallel', 'deflation']
            best['algorithm'] = comparison_generation.compare_mixing_values(model, data,
                                                                            'algorithm',
                                                                            values)
            comparison_generation.update_model(model, best)
            best['max_iter'] = comparison_generation.compare_mixing_numbers(model, data,
                                                                            'max_iter',
                                                                            intervals=10,
                                                                            interval_size=20,
                                                                            start_index=20)
            comparison_generation.update_model(model, best)
            io_manager.save_best(best, best_name)
            io_manager.update_convergence(best, best_name)
        comparison_generation.plot_reduction(model, data)
        training_features = data['training_features']
        test_features = data['test_features']
        new_training_features = pd.DataFrame(model.transform(training_features))
        new_test_features = pd.DataFrame(model.transform(test_features))
        print("For ICA, how kurtotic are the distributions?")
        model_name = type(model).__name__
        print(comparison_generation.create_title(data, model_name) \
              + str(scipy.stats.kurtosis(new_training_features)))
    except Exception as ex:
        print(ex)
        traceback.print_exc()
    return new_training_features, new_test_features
