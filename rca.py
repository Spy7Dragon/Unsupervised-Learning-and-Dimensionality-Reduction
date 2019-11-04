from sklearn import *

import comparison_generation
import io_manager
import pandas as pd
import traceback


def perform_rca(data):
    new_training_features = None
    new_test_features = None
    try:
        model = random_projection.GaussianRandomProjection(random_state=7)
        best_name = comparison_generation.create_attribute_storage_name(data, model)
        best = io_manager.load_best(best_name)
        comparison_generation.update_model(model, best)
        if not io_manager.is_converged(best, best_name):
            best['n_components'] = comparison_generation.compare_cluster_numbers(model, data,
                                                                                 'n_components',
                                                                                 intervals=5,
                                                                                 interval_size=1,
                                                                                 start_index=2)
            comparison_generation.update_model(model, best)
            io_manager.save_best(best, best_name)
            io_manager.update_convergence(best, best_name)
        comparison_generation.plot_reduction(model, data)
        training_features = data['training_features']
        test_features = data['test_features']
        new_training_features = pd.DataFrame(model.transform(training_features))
        new_test_features = pd.DataFrame(model.transform(test_features))
    except Exception as ex:
        print(ex)
        traceback.print_exc()
    return new_training_features, new_test_features
