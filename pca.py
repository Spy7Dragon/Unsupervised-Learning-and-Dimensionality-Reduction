from sklearn import *

import comparison_generation
import io_manager
import pandas as pd
import traceback


def perform_pca(data):
    new_training_features = None
    new_test_features = None
    try:
        model = decomposition.PCA(random_state=7)
        best_name = comparison_generation.create_attribute_storage_name(data, model)
        best = io_manager.load_best(best_name)
        comparison_generation.update_model(model, best)
        if not io_manager.is_converged(best, best_name):
            best['n_components'] = comparison_generation.compare_explained_variance_numbers(model, data,
                                                                                            'n_components',
                                                                                            intervals=6,
                                                                                            interval_size=1,
                                                                                            start_index=2)
            comparison_generation.update_model(model, best)
            values = [False, True]
            best['whiten'] = comparison_generation.compare_explained_variance_values(model, data,
                                                                                     'whiten',
                                                                                     values)
            comparison_generation.update_model(model, best)
            values = ['auto', 'full', 'arpack', 'randomized']
            best['svd_solver'] = comparison_generation.compare_explained_variance_values(model, data,
                                                                                         'svd_solver',
                                                                                         values)
            comparison_generation.update_model(model, best)
            io_manager.save_best(best, best_name)
            io_manager.update_convergence(best, best_name)
        comparison_generation.plot_reduction(model, data)
        training_features = data['training_features']
        test_features = data['test_features']
        new_training_features = pd.DataFrame(model.transform(training_features))
        new_test_features = pd.DataFrame(model.transform(test_features))
        model_name = comparison_generation.get_model_name(model)
        title = comparison_generation.create_title(data, model_name)
        print(title)
        print(model.explained_variance_)
    except Exception as ex:
        print(ex)
        traceback.print_exc()
    return new_training_features, new_test_features
