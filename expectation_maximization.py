from sklearn import *
import comparison_generation
import io_manager
import pandas as pd
import traceback


def perform_expectation_maximization(data):
    new_training_features = None
    new_test_features = None
    try:
        model = mixture.GaussianMixture(random_state=7, reg_covar=1e10)
        best_name = comparison_generation.create_attribute_storage_name(data, model)
        best = io_manager.load_best(best_name)
        comparison_generation.update_model(model, best)
        if not io_manager.is_converged(best, best_name):
            best['n_clusters'] = comparison_generation.compare_mean_numbers(model, data,
                                                                            'n_components',
                                                                            intervals=5,
                                                                            interval_size=1,
                                                                            start_index=2)
            comparison_generation.update_model(model, best)
            best['reg_covar'] = comparison_generation.compare_mean_numbers(model, data, 'reg_covar',
                                                                           intervals=10,
                                                                           interval_size=1e-7,
                                                                           start_index=1e-7)
            comparison_generation.update_model(model, best)
            io_manager.update_convergence(best, best_name)
            io_manager.save_best(best, best_name)
        comparison_generation.plot_reduction(model, data)
        training_features = data['training_features']
        training_classes = data['training_classes']
        test_features = data['test_features']
        new_training_features = pd.DataFrame(model.sample(len(training_features))[0])
        new_test_features = pd.DataFrame(model.sample(len(test_features))[0])
    except Exception as ex:
        print(ex)
        traceback.print_exc()
    return new_training_features, new_test_features
