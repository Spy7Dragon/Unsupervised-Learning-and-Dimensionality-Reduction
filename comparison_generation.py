import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
import pandas as pd
import traceback
from timeit import default_timer as timer

colors = ['red', 'orange', 'green', 'blue', 'indigo', 'violet']


def compare_cluster_values(model, data, attribute, values):
    training_features = data['training_features']
    training_classes = data['training_classes']
    test_features = data['test_features']
    test_classes = data['test_classes']
    model_name = type(model).__name__
    # print("Perform training for " + attribute + " on " + model_name)
    for i in range(0, len(values)):
        value = values[i]
        scatters = []
        try:
            setattr(model, attribute, value)
            model.fit(training_features)
            feature_sets = itertools.combinations(training_features.columns.tolist(), 2)
            for feature_set in feature_sets:
                feature1 = feature_set[0]
                feature2 = feature_set[1]
                for i in range(model.n_clusters):
                    color = colors[i]
                    scatter = plt.scatter(training_features.loc[model.labels_ == i, feature1],
                                          training_features.loc[model.labels_ == i, feature2],
                                          s=1, c=color, label='cluster' + str(i))
                    scatters.append(scatter)
                plt.legend()
                plt.xlabel(feature1)
                plt.ylabel(feature2)
                max_int = np.iinfo(np.int64).max
                minx = training_features.loc[training_features[feature1] != max_int][feature1].min(skipna=True)
                maxx = training_features.loc[training_features[feature1] != max_int][feature1].max(skipna=True)
                plt.xlim([minx, maxx])
                miny = training_features.loc[training_features[feature2] != max_int][feature2].min(skipna=True)
                maxy = training_features.loc[training_features[feature2] != max_int][feature2].max(skipna=True)
                plt.ylim([miny, maxy])
                title = create_title(data, model_name, attribute, value, feature1, feature2)
                plt.title(title)
                directory = data['directory']
                plt.savefig(directory + '/' + title + '.png')
                plt.close()
        except Exception as ex:
            print(str(ex) + ' in ' + model_name + ' using ' + attribute \
                  + ' of ' + str(value))
            traceback.print_exc()
        finally:
            for scatter in scatters:
                scatter.remove()


def display_clusters(model, data):
    training_features = data['training_features']
    training_classes = data['training_classes'].iloc[:, 0]
    test_features = data['test_features']
    test_classes = data['test_classes'].iloc[:, 0]
    model_name = type(model).__name__
    # print("Perform training on " + model_name)
    try:
        model.fit(training_features)
        feature_sets = itertools.combinations(training_features.columns.tolist(), 2)
        for feature_set in feature_sets:
            scatters = []
            try:
                feature1 = feature_set[0]
                feature2 = feature_set[1]
                scatter = plt.scatter(training_features.loc[training_classes.values == True][feature1],
                                      training_features.loc[training_classes.values == True][feature2],
                                      s=1, facecolors='black', label='True')
                scatters.append(scatter)
                scatter = plt.scatter(training_features.loc[training_classes.values == False][feature1],
                                      training_features.loc[training_classes.values == False][feature2],
                                      s=1, facecolors='gray', label='False')
                scatters.append(scatter)
                plt.legend()
                plt.xlabel(feature1)
                plt.ylabel(feature2)
                max_int = np.iinfo(np.int64).max
                minx = training_features.loc[training_features[feature1] != max_int][feature1].min(skipna=True)
                if np.isnan(minx):
                    minx = 0
                maxx = training_features.loc[training_features[feature1] != max_int][feature1].max(skipna=True)
                if np.isnan(maxx):
                    maxx = max_int
                plt.xlim([minx, maxx])
                miny = training_features.loc[training_features[feature2] != max_int][feature2].min(skipna=True)
                if np.isnan(miny):
                    miny = 0
                maxy = training_features.loc[training_features[feature2] != max_int][feature2].max(skipna=True)
                if np.isnan(maxy):
                    maxy = max_int
                plt.ylim([miny, maxy])
                title = create_title(data, model_name, feature1=feature1, feature2=feature2)
                plt.title(title)
                directory = data['directory']
                plt.savefig(directory + '/' + title + '.png')
                plt.close()
            except Exception as ex:
                print(str(ex) + ' in ' + model_name)
                traceback.print_exc()
            finally:
                for scatter in scatters:
                    scatter.remove()
    except Exception as ex:
        print(str(ex) + ' in ' + model_name)
        traceback.print_exc()


def compare_cluster_numbers(model, data, attribute, intervals, interval_size=20, start_index=0):
    training_features = data['training_features']
    training_classes = data['training_classes']
    test_features = data['test_features']
    test_classes = data['test_classes']
    model_name = type(model).__name__
    best_inertia = float('inf')
    best_training_time = float('inf')
    best_value = start_index
    # print("Perform training for " + attribute + " on " + model_name)
    for i in range(intervals):
        value = i * interval_size + start_index
        try:
            setattr(model, attribute, value)
            start = timer()
            model.fit(training_features)
            end = timer()
            training_time = end - start
            feature_sets = itertools.combinations(training_features.columns.tolist(), 2)
            for feature_set in feature_sets:
                scatters = []
                try:
                    feature1 = feature_set[0]
                    feature2 = feature_set[1]
                    length = 2
                    if hasattr(model, 'n_components'):
                        length = model.n_components
                    elif hasattr(model, 'n_clusters'):
                        length = model.n_clusters
                    for i in range(length):
                        color = colors[i]
                        scatter = plt.scatter(training_features.loc[model.labels_ == i, feature1],
                                              training_features.loc[model.labels_ == i, feature2],
                                              s=1, c=color, label='cluster' + str(i))
                        scatters.append(scatter)
                    plt.legend()
                    plt.xlabel(feature1)
                    plt.ylabel(feature2)
                    max_int = np.iinfo(np.int64).max
                    minx = training_features.loc[training_features[feature1] != max_int][feature1].min(skipna=True)
                    if np.isnan(minx):
                        minx = 0
                    maxx = training_features.loc[training_features[feature1] != max_int][feature1].max(skipna=True)
                    if np.isnan(maxx):
                        maxx = max_int
                    plt.xlim([minx, maxx])
                    miny = training_features.loc[training_features[feature2] != max_int][feature2].min(skipna=True)
                    if np.isnan(miny):
                        miny = 0
                    maxy = training_features.loc[training_features[feature2] != max_int][feature2].max(skipna=True)
                    if np.isnan(maxy):
                        maxy = max_int
                    plt.ylim([miny, maxy])
                    title = create_title(data, model_name, attribute, value, feature1, feature2)
                    plt.title(title)
                    directory = data['directory']
                    plt.savefig(directory + '/' + title + '.png')
                    for scatter in scatters:
                        scatter.remove()
                    plt.close()
                except Exception as ex:
                    print(str(ex) + ' in ' + model_name + ' using ' + attribute \
                          + ' of ' + str(value))
                    traceback.print_exc()
            if hasattr(model, 'inertia_'):
                inertia = model.inertia_
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_value = value
            else:
                if training_time < best_training_time:
                    best_training_time = training_time
                    best_value = value
        except Exception as ex:
            print(str(ex) + ' in ' + model_name + ' using ' + attribute \
                  + ' of ' + str(value))
            traceback.print_exc()

    return best_value


def compare_error_numbers(model, data, attribute, intervals, interval_size=20, start_index=0):
    training_features = data['training_features']
    training_classes = data['training_classes']
    test_features = data['test_features']
    test_classes = data['test_classes']
    model_name = type(model).__name__
    section_scores = []
    time_data = []
    # print("Perform training for " + attribute + " on " + model_name)
    for i in range(intervals):
        value = i * interval_size + start_index
        try:
            setattr(model, attribute, value)
            start = timer()
            model.fit(training_features, training_classes)
            end = timer()
            training_time = end - start
            start = timer()
            predicted_training_classes = model.predict(training_features)
            end = timer()
            classification_time = end - start
            time_data.append([training_time, classification_time])

            predicted_test_classes = model.predict(test_features)
            training_score = accuracy_score(training_classes, predicted_training_classes)
            training_error = 1.0 - training_score
            test_score = accuracy_score(test_classes, predicted_test_classes)
            test_error = 1.0 - test_score
            section_scores.append([value, training_error, test_error])
        except Exception as ex:
            print(str(ex) + ' in ' + model_name + ' using ' + attribute \
                  + ' of ' + str(value))
            traceback.print_exc()
    title = create_title(data, model_name, attribute)
    plot_frame = pd.DataFrame(section_scores, columns=[attribute, 'Training Error', 'Test Error'])
    graph = plot_frame.plot(x=attribute, y=['Training Error', 'Test Error'],
                            title=title)
    graph.set_xlabel(attribute)
    graph.set_ylabel('Error')

    plt.ylim(0.0, 0.5)
    directory = data['directory']
    plt.savefig(directory + '/' + title + '.png')
    plt.close()

    time_table = pd.DataFrame(time_data,
                              columns=['Training Time', 'Classification Time'])
    table_directory = data['table_directory']
    time_table.to_csv(table_directory + '/' + title + '-time.csv')


def compare_mean_values(model, data, attribute, values):
    training_features = data['training_features']
    training_classes = data['training_classes']
    test_features = data['test_features']
    test_classes = data['test_classes']
    model_name = type(model).__name__
    # print("Perform training for " + attribute + " on " + model_name)
    for i in range(0, len(values)):
        value = values[i]
        scatters = []
        try:
            setattr(model, attribute, value)
            model.fit(training_features)
            feature_list = training_features.columns.tolist()
            feature_sets = itertools.combinations(feature_list, 2)
            for feature_set in feature_sets:
                feature1 = feature_set[0]
                feature2 = feature_set[1]
                for index in range(len(model.means_)):
                    color = colors[index]
                    scatter = plt.scatter(model.means_[index, feature1],
                                          model.means_[index, feature2],
                                          s=3, c=color, label='cluster' + str(index))
                    scatters.append(scatter)
                plt.legend()
                plt.xlabel(feature1)
                plt.ylabel(feature2)
                max_int = np.iinfo(np.int64).max
                minx = training_features.loc[training_features[feature1] != max_int][feature1].min(skipna=True)
                maxx = training_features.loc[training_features[feature1] != max_int][feature1].max(skipna=True)
                plt.xlim([minx, maxx])
                miny = training_features.loc[training_features[feature2] != max_int][feature2].min(skipna=True)
                maxy = training_features.loc[training_features[feature2] != max_int][feature2].max(skipna=True)
                plt.ylim([miny, maxy])
                title = create_title(data, model_name, attribute, value, feature1, feature2)
                plt.title(title)
                directory = data['directory']
                plt.savefig(directory + '/' + title + '.png')
                plt.close()
        except Exception as ex:
            print(str(ex) + ' in ' + model_name + ' using ' + attribute \
                  + ' of ' + str(value))
            traceback.print_exc()
        finally:
            for scatter in scatters:
                scatter.remove()


def compare_mean_numbers(model, data, attribute, intervals, interval_size=20, start_index=0):
    training_features = data['training_features']
    training_classes = data['training_classes']
    training_features = training_features.loc[:, (training_features != 0).any(axis=0)]
    training_features.dropna(inplace=True)
    used_columns = training_features.columns.tolist()
    test_features = data['test_features']
    test_classes = data['test_classes']
    test_features = test_features.loc[:, used_columns]
    test_features.dropna(inplace=True)
    model_name = type(model).__name__
    best_value = start_index
    best_training_time = float('inf')
    # print("Perform training for " + attribute + " on " + model_name)
    for i in range(intervals):
        value = i * interval_size + start_index
        scatters = []
        try:
            setattr(model, attribute, value)
            start = timer()
            model.fit(training_features)
            end = timer()
            training_time = end - start
            feature_sets = itertools.combinations(range(len(used_columns)), 2)
            for feature_set in feature_sets:
                feature1 = feature_set[0]
                feature2 = feature_set[1]
                for index in range(len(model.means_)):
                    color = colors[index]
                    scatter = plt.scatter(model.means_[index, feature1],
                                          model.means_[index, feature2],
                                          s=5, c=color, label='cluster' + str(index))
                    scatters.append(scatter)
                plt.legend()
                plt.xlabel(feature1)
                plt.ylabel(feature2)
                feature1 = used_columns[feature1]
                feature2 = used_columns[feature2]
                max_int = np.iinfo(np.int64).max
                minx = training_features.loc[training_features[feature1] != max_int][feature1].min(skipna=True)
                maxx = training_features.loc[training_features[feature1] != max_int][feature1].max(skipna=True)
                plt.xlim([minx, maxx])
                miny = training_features.loc[training_features[feature2] != max_int][feature2].min(skipna=True)
                maxy = training_features.loc[training_features[feature2] != max_int][feature2].max(skipna=True)
                plt.ylim([miny, maxy])
                title = create_title(data, model_name, attribute, value, feature1, feature2)
                plt.title(title)
                directory = data['directory']
                plt.savefig(directory + '/' + title + '.png')
                for scatter in scatters[:]:
                    try:
                        scatter.remove()
                    except Exception:
                        pass
                plt.close()
                if training_time < best_training_time:
                    best_training_time = training_time
                    best_value = value
        except Exception as ex:
            print(str(ex) + ' in ' + model_name + ' using ' + attribute \
                  + ' of ' + str(value))
            traceback.print_exc()
    return best_value


def compare_explained_variance_numbers(model, data, attribute, intervals, interval_size=20, start_index=0):
    training_features = data['training_features']
    training_classes = data['training_classes']
    test_features = data['test_features']
    test_classes = data['test_classes']
    model_name = type(model).__name__
    best_mean_explained_variance = 0.0
    best_value = start_index
    # print("Perform training for " + attribute + " on " + model_name)
    for i in range(intervals):
        value = i * interval_size + start_index
        try:
            setattr(model, attribute, value)
            new_data = model.fit_transform(training_features)
            mean_explained_variance = np.sum(model.explained_variance_ratio_)
            if mean_explained_variance > best_mean_explained_variance \
                    and best_mean_explained_variance < 1.0:
                best_mean_explained_variance = mean_explained_variance
                best_value = value
        except Exception as ex:
            print(str(ex) + ' in ' + model_name + ' using ' + attribute \
                  + ' of ' + str(value))
            traceback.print_exc()
    setattr(model, attribute, best_value)
    return best_value


def compare_explained_variance_values(model, data, attribute, values):
    training_features = data['training_features']
    training_classes = data['training_classes']
    test_features = data['test_features']
    test_classes = data['test_classes']
    model_name = type(model).__name__
    best_mean_explained_variance = 0.0
    best_value = values[0]
    # print("Perform training for " + attribute + " on " + model_name)
    for i in range(0, len(values)):
        value = values[i]
        try:
            setattr(model, attribute, value)
            new_data = model.fit_transform(training_features)
            mean_explained_variance = np.sum(model.explained_variance_ratio_)
            if mean_explained_variance > best_mean_explained_variance \
                    and best_mean_explained_variance < 1.0:
                best_mean_explained_variance = mean_explained_variance
                best_value = value
        except Exception as ex:
            print(str(ex) + ' in ' + model_name + ' using ' + attribute \
                  + ' of ' + str(value))
            traceback.print_exc()
    setattr(model, attribute, best_value)
    return best_value


def compare_mixing_numbers(model, data, attribute, intervals, interval_size=20, start_index=0):
    training_features = data['training_features']
    training_classes = data['training_classes']
    test_features = data['test_features']
    test_classes = data['test_classes']
    model_name = type(model).__name__
    best_mixing_sum = 0.0
    best_value = start_index
    # print("Perform training for " + attribute + " on " + model_name)
    for i in range(intervals):
        value = i * interval_size + start_index
        try:
            setattr(model, attribute, value)
            new_data = model.fit_transform(training_features)
            mixing_sum = np.sum(model.mixing_)
            if mixing_sum > best_mixing_sum \
                    and best_mixing_sum < 1.0:
                best_mixing_sum = mixing_sum
                best_value = value
        except Exception as ex:
            print(str(ex) + ' in ' + model_name + ' using ' + attribute \
                  + ' of ' + str(value))
            traceback.print_exc()
    setattr(model, attribute, best_value)
    return best_value


def compare_mixing_values(model, data, attribute, values):
    training_features = data['training_features']
    training_classes = data['training_classes']
    test_features = data['test_features']
    test_classes = data['test_classes']
    model_name = type(model).__name__
    best_mixing_sum = 0.0
    best_value = values[0]
    # print("Perform training for " + attribute + " on " + model_name)
    for i in range(0, len(values)):
        value = values[i]
        try:
            setattr(model, attribute, value)
            new_data = model.fit_transform(training_features)
            mixing_sum = np.sum(model.mixing_)
            if mixing_sum > best_mixing_sum \
                    and best_mixing_sum < 1.0:
                best_mixing_sum = mixing_sum
                best_value = value
        except Exception as ex:
            print(str(ex) + ' in ' + model_name + ' using ' + attribute \
                  + ' of ' + str(value))
            traceback.print_exc()
    setattr(model, attribute, best_value)
    return best_value


def plot_reduction(model, data):
    training_features = data['training_features']
    training_classes = data['training_classes']
    test_features = data['test_features']
    model.fit(training_features)
    if hasattr(model, 'transform'):
        new_training_features = pd.DataFrame(model.transform(training_features))
    else:
        new_training_features = pd.DataFrame(model.sample(len(training_features))[0])
    length = 2
    if hasattr(model, 'n_components'):
        length = model.n_components
    elif hasattr(model, 'n_clusters'):
        length = model.n_clusters
    feature_sets = itertools.combinations(range(length), 2)
    model_name = type(model).__name__
    for feature_set in feature_sets:
        feature1 = feature_set[0]
        feature2 = feature_set[1]
        scatter = plt.scatter(new_training_features.iloc[:, feature1],
                              new_training_features.iloc[:, feature2],
                              s=1, color='black', label='points')
        plt.legend()
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        max_int = np.iinfo(np.int64).max
        minx = new_training_features.loc[new_training_features[feature1] != max_int][feature1].min(skipna=True)
        if np.isnan(minx):
            minx = 0
        maxx = new_training_features.loc[new_training_features[feature1] != max_int][feature1].max(skipna=True)
        if np.isnan(maxx):
            maxx = max_int
        plt.xlim([minx, maxx])
        miny = new_training_features.loc[new_training_features[feature2] != max_int][feature2].min(skipna=True)
        if np.isnan(miny):
            miny = 0
        maxy = new_training_features.loc[new_training_features[feature2] != max_int][feature2].max(skipna=True)
        if np.isnan(maxy):
            maxy = max_int
        plt.ylim([miny, maxy])
        title = create_title(data, model_name, feature1=feature1, feature2=feature2)
        plt.title(title)
        directory = data['directory']
        plt.savefig(directory + '/' + title + '-reduction.png')
        scatter.remove()
        plt.close()


def get_model_name(model):
    return type(model).__name__


def create_title(data, model_name, attribute=None, value=None, feature1=None, feature2=None):
    title = ""
    if 'dimensionality_reduction' in data and data['dimensionality_reduction'] is not None:
        title += data['dimensionality_reduction'] + '-'
    if 'clusterer' in data and data['clusterer'] is not None:
        title += data['clusterer'] + '-'
    title += model_name
    if attribute is not None:
        title += '-' + attribute
    if value is not None:
        title += '-' + str(value)
    if feature1 is not None and feature2 is not None:
        title += '-' + str(feature1) + '-vs-' + str(feature2)
    return title


def create_attribute_storage_name(data, model):
    directory = data['directory']
    model_name = type(model).__name__
    name = directory + '-' + model_name
    if 'dimensionality_reduction' in data and data['dimensionality_reduction'] is not None:
        name += '-' + data['dimensionality_reduction']
    if 'clusterer' in data and data['clusterer'] is not None:
        name += '-' + data['clusterer']
    return name


def update_model(model, best):
    for attribute in best:
        setattr(model, attribute, best[attribute])
