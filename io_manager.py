import pickle


def load_best(name):
    best = {}
    try:
        with open('best_attributes/' + name + '.pkl', 'rb') as file:
            best = pickle.load(file)
    except Exception as ex:
        print(ex)
    return best


def save_best(best, name):
    try:
        with open('best_attributes/' + name + '.pkl', 'wb') as file:
            pickle.dump(best, file, pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(ex)


def update_convergence(best, best_name):
    convergence = {}
    try:
        with open('best_attributes/convergence.pkl', 'rb') as file:
            convergence = pickle.load(file)
    except Exception as ex:
        print(ex)
    old_best = {}
    if best_name in convergence:
        old_best = convergence[best_name]['best']
    else:
        convergence[best_name] = {}
    converged = old_best == best
    convergence[best_name]['best'] = best
    convergence[best_name]['converged'] = converged
    try:
        with open('best_attributes/convergence.pkl', 'wb') as file:
            pickle.dump(convergence, file, pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(ex)


def is_converged(best, best_name):
    convergence = {}
    try:
        with open('best_attributes/convergence.pkl', 'rb') as file:
            convergence = pickle.load(file)
    except Exception as ex:
        print(ex)
    old_best = None
    if best_name in convergence:
        old_best = convergence[best_name]['best']
    else:
        convergence[best_name] = {}
    same = old_best == best
    converged = False
    if best_name in convergence and 'converged' in convergence[best_name]:
        converged = convergence[best_name]['converged']
    converged = same and converged and len(best) > 0
    if converged:
        print(best_name + '-' + str(convergence[best_name]['best']))
    return converged
