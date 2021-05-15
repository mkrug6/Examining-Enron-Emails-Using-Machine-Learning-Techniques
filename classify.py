from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tester import test_classifier
from sklearn.grid_search import GridSearchCV
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

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


def try_classifiers(data, features_list):
    """
    Tries different classifiers and then chooses the best one
    """

    data = featureFormat(data, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    print('Trying AdaBoost')
    clf_ab = AdaBoostClassifier(DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=2,
        class_weight='balanced'),
        n_estimators=50,
        learning_rate=.8)

    clf_ab_grid_search = GridSearchCV(clf_ab, {})
    clf_ab_grid_search.fit(features_train, labels_train)
    clf_ab_grid_search.best_estimator_
    test_classifier(clf_ab_grid_search, data, features_list)

    print('Trying GaussianNB')
    clf_gb = GaussianNB()
    clf_gb_grid_search = GridSearchCV(clf_gb, {})
    clf_gb_grid_search.fit(features_train, labels_train)
    clf_gb_grid_search.best_estimator_

    print('Trying SVC')
    
    clf_svc = SVC(kernel='linear', max_iter=1000)
    clf_svc_grid_search = GridSearchCV(clf_svc, {})
    clf_svc_grid_search.fit(features_train, labels_train)
    clf_svc_grid_search.best_estimator_
   
    # Return the one which perform the best
    return clf_ab_grid_search
