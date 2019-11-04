from sklearn import *
import io_manager

import comparison_generation
import traceback


def perform_neural_network(data):
    try:
        model = neural_network.MLPClassifier(hidden_layer_sizes=4, random_state=7,
                                             early_stopping=True)
        best_name = comparison_generation.create_attribute_storage_name(data, model)
        best = io_manager.load_best(best_name)
        comparison_generation.update_model(model, best)
        if not io_manager.is_converged(best, best_name):
            comparison_generation.compare_error_numbers(model, data, 'random_state', intervals=10,
                                                        interval_size=1,
                                                        start_index=1)
            comparison_generation.update_model(model, best)
            io_manager.save_best(best, best_name)
            io_manager.update_convergence(best, best_name)
    except Exception as ex:
        print(ex)
        traceback.print_exc()
