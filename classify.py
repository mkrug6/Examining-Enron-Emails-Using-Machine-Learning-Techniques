from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tester import test_classifier
from sklearn.grid_search import GridSearchCV

def evaluate_clf(grid_search, features, labels, params, iters=100):
    """
    Evaluate a classifier
    """
    acc = []
    pre = []
    recall = []

    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)]
        pre = pre + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
    print "accuracy: {}".format(mean(acc))
    print "precision: {}".format(mean(pre))
    print "recall:    {}".format(mean(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))


def try_classifiers(data, features_list, tune=False):
    """
    Tries different classifiers and then chooses the best one
    """

    print('Trying GaussianNB')
    clf_gb = GaussianNB()
    clf_gb_grid_search = GridSearchCV(clf_gb, {})
    test_classifier(clf_gb_grid_search, data, features_list)
    # evaluate_clf(nb_grid_search, features, labels, {})

    print('Trying RandomForest')
    clf_rf = RandomForestClassifier()
    clf_rf_grid_search = GridSearchCV(clf_rf, {})
    test_classifier(clf_rf_grid_search, data, features_list)
    # evaluate_clf(nb_grid_search, features, labels, {})

    if tune:
        print('Trying Tuned RandomForest')
        clf_rft = RandomForestClassifier()
        clf_rft_grid_search = GridSearchCV(clf_rft, {'n_estimators': [4, 5, 10, 500]})
        test_classifier(clf_rft_grid_search, data, features_list)
        # evaluate_clf(nb_grid_search, features, labels, {})

    print('Trying SVC...')
    clf_svc = SVC(kernel='linear', max_iter=1000)
    clf_svc_grid_search = GridSearchCV(clf_svc, {})
    test_classifier(clf_svc_grid_search, data, features_list)
    # evaluate_clf(nb_grid_search, features, labels, {})

    if tune:
        print('Trying Tuned SVC')
        param_grid = {'C': [1, 50, 100, 1000],
                      'gamma': [0.5, 0.1, 0.01],
                      'degree': [1, 2],
                      'kernel': ['rbf', 'poly', 'linear'],
                      'max_iter': [1, 100, 1000]}
        clf_svct = SVC(kernel='linear', max_iter=1000)
        clf_svct_grid_search = GridSearchCV(clf_svct, param_grid)
        test_classifier(clf_svct_grid_search, data, features_list)
        # evaluate_clf(nb_grid_search, features, labels, {})

    # Return the one which perform the best
    return clf_gb_grid_search
