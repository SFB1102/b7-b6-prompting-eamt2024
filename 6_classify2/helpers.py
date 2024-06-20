"""
20 Sept 2023
classifier helpers
"""

import numpy as np
import os
import sys

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import SelectKBest
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from collections import defaultdict

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.utils import shuffle


def featureselection(x, labels, features=None):
    # Feature selection
    if set(labels) > 10:  # we are dealing with a regression task
        print('Using F-value between label/feature for regression tasks.')
        ff = SelectKBest(score_func=f_regression, k=features).fit(x, labels)
        newdata = SelectKBest(score_func=f_regression, k=features).fit_transform(x, labels)
    else:  # this is a classification task: using one-way (=one-target) ANOVA for paramentric data ()
        print('Using F-value between label/feature for classification tasks.')
        ff = SelectKBest(score_func=f_classif, k=features).fit(x, labels)
        newdata = SelectKBest(score_func=f_classif, k=features).fit_transform(x, labels)
    top_ranked_features = sorted(enumerate(ff.scores_), key=lambda y: y[1], reverse=True)[:features]
    # print(top_ranked_features) ## it is a list of tuples (11, 3930.587580730744)
    top_ranked_features_indices = [x[0] for x in top_ranked_features]
    return newdata, top_ranked_features_indices


def my_spearman_r(trues, predictions):
    return spearmanr(trues, predictions)[0]


def recursive_elimination(X, y, feats=None, algo=None, cv=5):  # SVR or RF
    if algo == 'SVM':
        clf = SVC(kernel="linear", C=1.0)
        scoring = 'f1_macro'
    elif algo == 'SVR':
        clf = SVR(kernel="linear", C=1.0)
        scoring = make_scorer(my_spearman_r, greater_is_better=True)
        # scoring = 'r2'  # neg_root_mean_squared_error, pearson? spearman
    else:
        clf = None
        scoring = None
        print('Choose algo (SVC or SVR) for recursive feature elimination')
    clf.fit(X, y)

    if feats == -1:
        # RFECV only uses CV to select the optimal number of features to select.
        # It then uses regular RFE (on the entire set) to actually preform selection.
        print(f'Running feature selection with {algo} in {cv}-fold cross-validation setting')
        selector = RFECV(clf, step=1, cv=cv, n_jobs=-1, min_features_to_select=3, scoring=scoring)
    else:
        selector = RFE(clf, n_features_to_select=feats, step=1)

    selector = selector.fit(X, y)
    print(selector)
    top_idx = [x[0] for x in enumerate(selector.support_) if x[1]]

    return top_idx


def get_xy_best(training_set, category=None, features=None, muted=None, select_mode=None, scaling=1, algo=None, cv=10):
    y0 = training_set.loc[:, category].values
    fns = training_set['iid'].tolist()
    # drop remaining meta
    if muted:
        try:
            training_set = training_set.drop(['iid', 'ttype', 'lang'] + muted, axis=1)
        except KeyError:
            training_set = training_set.drop(['iid', 'ttype', 'lang'], axis=1)
    else:
        training_set = training_set.drop(['iid', 'ttype', 'lang'], axis=1)
    print(training_set.head())
    print(training_set.shape)

    print(f'Number of input features in get_xy_best: {training_set.shape[1]}')
    sc = StandardScaler()
    if isinstance(select_mode, list):
        new_df = training_set[select_mode]
        if scaling:
            print(f'===StandardScaler() for preselect lang-specific feats===')
            x0 = sc.fit_transform(training_set)
        else:
            x0 = training_set
        featurelist = select_mode
        print(x0.head())
        exit()
    elif features or select_mode == 'RFECV':
        top_feat = None
        if select_mode == 'RFE' or select_mode == 'RFECV':
            if scaling:
                print(f'===StandardScaler() BEFORE {select_mode} selection!===')
                # centering and scaling for each feature
                # transform your data such that its distribution will have a mean value 0 and standard deviation of 1
                # to meet the assumption that all features are centered around 0 and have variance in the same order
                # each value will have the sample mean subtracted, and then divided by the StD of the distribution
                myfeats = training_set.columns.tolist()
                training_set[myfeats] = sc.fit_transform(training_set[myfeats])
            else:
                print('Using raw unscaled X data')

            top_feat = recursive_elimination(training_set, y0, feats=features, algo=algo, cv=cv)
            x0 = training_set.iloc[:, top_feat].values
            new_df = training_set.iloc[:, top_feat]
        else:
            x0 = None
            new_df = None
            print('Choose a feature selection algorithm')

        print('Data (X for SVM) after feature selection:', x0.shape)
        featurelist = [training_set.keys()[i] for i in top_feat]
    else:
        # no feature selection
        x0 = training_set.values
        new_df = training_set
        print('Data (X for SVM), no feature selection:', x0.shape)
        featurelist = None  # [i for i in new_df.columns.tolist() if i not in metas]

    return x0, y0, featurelist, fns, new_df


def get_experimental_slice(_df=None, _lang=None, my_feats=None):
    _df = _df[_df.lang == _lang]
    # if there is a mismatch in feature names (addit is called additive), I lose the entire rewritten class here
    _df = _df.dropna(subset=my_feats)
    y = _df['ttype'].values

    this_df = _df[my_feats]
    _x = this_df.values

    return _x, y, _df


def make_dirs(my_dirs=None):
    for i in my_dirs:
        os.makedirs(i, exist_ok=True)


class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")  # overwrite, don't "a" append

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def crossvalidated_svm(x, y, algo='SVM', splitter=None, class_weight='balanced', rand=None, run=None):
    print(f'Features: {x.shape[1]}')
    clf = None
    if algo == 'SVM':
        if run == 'default_linear':
            clf = SVC(kernel='linear', C=1.0, random_state=rand, class_weight=class_weight, probability=True)
        elif run == 'default_rbf':
            # default sklearn
            clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=rand, class_weight=class_weight,
                      probability=True)
        elif run == 'tuned_poly':
            print('\n===tuned_poly====\n')
            clf = SVC(kernel='poly', C=100, degree=2, gamma=0.1, random_state=rand,
                      class_weight=class_weight, probability=True)
        else:
            print('\n===== DEFAULT LINEAR =======')
            print('I am not clear about SVM settings. Running with default linear kernel ...')
            clf = SVC(kernel='linear', C=1.0, random_state=rand, class_weight=class_weight, probability=True)

    elif algo == 'DUMMY':
        clf = DummyClassifier(strategy='stratified', random_state=rand)

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_scores = cross_validate(clf, x, y, cv=splitter, scoring=scoring, return_estimator=True, n_jobs=-1)

    scores_acc = cv_scores['test_accuracy']
    scores_preci = cv_scores['test_precision_macro']
    scores_recall = cv_scores['test_recall_macro']
    scores_f = cv_scores['test_f1_macro']

    classes_order = None
    if algo != 'DUMMY':
        for fold, model in enumerate(cv_scores['estimator']):
            if fold == 0:
                classes_order = model.classes_  # it is the same in all folds!
            else:
                classes_order = classes_order

    preds = cross_val_predict(clf, x, y, cv=splitter)

    predicted_probabilities = cross_val_predict(clf, x, y, cv=5, method='predict_proba')

    return scores_acc, scores_preci, scores_recall, scores_f, preds, classes_order, predicted_probabilities


# best_class_feats should be of size: features/2
def plot_weighted_scores(x, y, feature_names=None, colors=None, saveresto=None, savepicsto=None,
                         seed=None, run=None, verbose=None):
    # class_indicators_names = {k: [] for k in set(y)}
    # class_indicators_coef = {k: [] for k in set(y)}

    # shuffle for estimating weights on entire data, without cv
    X_shuffled, y_shuffled, = shuffle(x, y, random_state=seed)

    clf = SVC(C=1.0, kernel='linear', random_state=seed, class_weight='balanced')
    clf.fit(X_shuffled, y_shuffled)

    binary = clf.classes_
    weights_df = pd.DataFrame(zip(feature_names, abs(clf.coef_[0]), clf.coef_[0]),
                              columns=["feature", "abs_weight", "weight"]).sort_values("abs_weight",
                                                                                       ascending=False).reset_index(
        drop=True)
    # print(weights_df.head())
    # I need weights for all and for optimal feature sets
    if weights_df.shape[0] == 58:
        weights_df.to_csv(f'{saveresto}allfeats_{run}_{weights_df.shape[0]}feats.tsv', sep='\t', index=False)
    elif 15 <= weights_df.shape[0] < 58:
        weights_df.to_csv(f'{saveresto}optfeats_{run}_{weights_df.shape[0]}feats.tsv', sep='\t', index=False)
    else:
        pass

    # boxplots: classes separation
    scores = np.dot(x, clf.coef_.T)  # the result of multiplying a matrix by a vector is a vector!

    b0 = np.array(y) == binary[0]  # boolean or "mask" index arrays
    b1 = np.array(y) == binary[1]
    class0_scores = scores[b0]
    class1_scores = scores[b1]

    print(f'Number of instances in class0 ({binary[0]}): {class0_scores.shape[0]}, '
          f'class1 ({binary[1]}): {class1_scores.shape[0]}')

    color0 = colors[binary[0]]
    color1 = colors[binary[1]]

    if np.mean(class0_scores) < 0:
        neg_class_col = color0
        neg_class_name = binary[0]
        print(f'Class0 ({neg_class_name}) has negative average score (value*weight): {np.mean(class0_scores):.3f}')
    else:
        pos_class_col = color0
        pos_class_name = binary[0]
        print(f'Class0 ({pos_class_name}) has positive average score (value*weight): {np.mean(class0_scores):.3f}')

    if np.mean(class1_scores) < 0:
        neg_class_col = color1
        neg_class_name = binary[1]
        print(f'Class1 ({neg_class_name}) has negative average: {np.mean(class1_scores):.3f}')
    else:
        pos_class_col = color1
        pos_class_name = binary[1]
        print(f'Class1 ({pos_class_name}) has positive average: {np.mean(class1_scores):.3f}')

    # bar charts: features typical for each class, based on the negative or positive location of the boxplot?
    # print(coefs)
    feats_per_class = int(len(feature_names) / 2)
    top_positive_coefficients = np.argsort(clf.coef_[0])[-feats_per_class:]
    top_negative_coefficients = np.argsort(clf.coef_[0])[:feats_per_class]  # [::-1]

    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    feature_names = np.array(feature_names)

    plt.figure(figsize=(8, 8))
    # Set font properties globally
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'font.weight': 'normal'
    })

    colors_ = [neg_class_col if c < 0 else pos_class_col for c in clf.coef_[0][top_coefficients]]

    plt.bar(np.arange(2 * feats_per_class), clf.coef_[0][top_coefficients], color=colors_)
    plt.xticks(np.arange(2 * feats_per_class), feature_names[top_coefficients], rotation=45, ha='right')

    # creating a proxy artist specifically for adding to the legend
    labels = [i for i in list(colors.keys())]

    my_map = {'source': 'original', 'target': 'translated', 'rewritten': 'rewritten'}
    # my_map = {'source': 'original', 'rewritten': 'rewritten'}
    true_labels = [my_map[i] for i in labels]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, true_labels)

    plt.grid(color='darkgrey', linestyle='--', linewidth=0.3, alpha=0.5)

    print(run)

    level = run.split('_')[2].replace('seg', 'segment').replace('doc', 'document')

    if run.startswith('de'):
        plt.title(f'German, {level}: SVM weights')
    else:
        plt.title(f'English, {level}: SVM weights')
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    plt.savefig(f'{savepicsto}{run}-{len(feature_names)}feats.png')
    if verbose > 1:
        plt.show()

        # features weighted to indicate each class
        cat_keyed_feats = defaultdict(list)
        for c, f in zip(colors_, feature_names[top_coefficients]):
            cat = list(colors.keys())[list(colors.values()).index(c)]
            cat_keyed_feats[cat].append(f)

        for k, v in cat_keyed_feats.items():
            print(k, len(v), v)


def pca_this_lang(x=None, y=None, fns=None, save_name=None, lang=None, verbose=None):
    # transforming vectors
    print(x.shape)
    pca = PCA(n_components=2)
    x = pca.fit_transform(x)
    print(x.shape)

    settings = defaultdict()
    keys = ['aliases', 'names', 'cols', 'lines', 'points', 'edge']
    vals = [['source', 'target'],
            ['original', 'translated'],
            ['blue', 'red'],
            ['solid', 'solid'],
            ['.', 'x'],
            ['darkblue', 'darkred']]
    for k, vs in zip(keys, vals):
        settings[k] = vs

    fig, ax = plt.subplots(figsize=(12, 9))  # 4:3 ratio
    POINTSIZE = 40 # size of point labels
    FONTSIZE = 14
    COLOR = 'black'  # this is the color of the text, not of the datapoint
    XYTEXT = (6, 6)
    colored_items = []
    for idx, cat in enumerate(settings['aliases']):
        for i, _ in enumerate(x):
            if y[i] == cat:
                ax.scatter(*x[i], s=POINTSIZE, marker=settings['points'][idx],
                           # edgecolor=settings['edge'][idx],
                           color=settings['cols'][idx], alpha=0.7)
                if lang == 'en':
                    if ('source' in cat or 'target' in cat) and x[i][1] > 15:
                        # print(fns[i])
                        plt.annotate(fns[i],
                                     fontsize=FONTSIZE,
                                     color=COLOR,  # this is the color of the text, not of the datapoint
                                     xy=(x[i][0], x[i][1]),
                                     xytext=XYTEXT,
                                     # if you don't use this plt throws up a warning that it uses `textcoords` kwarg
                                     textcoords='offset points',
                                     ha='right',
                                     # this counterintuitively places the labels wrt the points (try: left, top)
                                     va='bottom')
                if lang == 'de':
                    if ('source' in cat or 'target' in cat) and x[i][1] > 10:
                        # print(fns[i])
                        plt.annotate(fns[i],
                                     fontsize=FONTSIZE,
                                     color=COLOR,  # this is the color of the text, not of the datapoint
                                     xy=(x[i][0], x[i][1]),
                                     xytext=XYTEXT,
                                     # if you don't use this plt throws up a warning that it uses `textcoords` kwarg
                                     textcoords='offset points',
                                     ha='right',
                                     # this counterintuitively places the labels wrt the points (try: left, top)
                                     va='bottom')
        colmarker = mlines.Line2D([], [], color=settings['cols'][idx], marker=settings['points'][idx], linestyle='None',
                                  markersize=5, label=settings['names'][idx])
        colored_items.append(colmarker)
    legend1 = plt.legend(handles=colored_items, fontsize=14, loc=1)  # bbox_to_anchor=(1, 1), loc='best'
    plt.gca().add_artist(legend1)

    # hide tick and tick label of the big axis
    plt.grid(color='darkgrey', linestyle='--', linewidth=0.3, alpha=0.5)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    if lang == 'en':
        plt.title('English', fontdict={'fontsize': 14})
    else:
        plt.title('German', fontdict={'fontsize': 14})

    ax.set_xlabel('PCA D1 values', fontsize=14)
    ax.set_ylabel('PCA D2 values', fontsize=14)
    # plt.title(f'based on PCA transform of {vect.upper()} representation', ha='center', fontsize=14)
    # plt.title(f'PCA transform of {vect.upper()} representation', ha='center', fontsize=14)

    string = f'Variance explained on D1: {pca.explained_variance_ratio_[0]:.2f}, D2: {pca.explained_variance_ratio_[1]:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if lang == 'en':
        plt.text(2, -5, string, fontsize=14, verticalalignment='top', bbox=props)
    else:
        plt.text(2, -6.5, string, fontsize=14, verticalalignment='top', bbox=props)

    if save_name:
        plt.savefig(save_name)
    if verbose > 1:
        plt.show()
