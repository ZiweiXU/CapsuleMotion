import numpy as np
import sklearn.cluster
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment


def knn_classify(data_train, data_test, label_train, label_test, nn=1):
    label_train = np.asarray(label_train)
    label_test = np.asarray(label_test)

    Xtr_Norm = preprocessing.normalize(data_train)
    Xte_Norm = preprocessing.normalize(data_test)

    # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
    knn = KNeighborsClassifier(n_neighbors=nn,
                               metric='cosine',
                               n_jobs=-1)
    knn.fit(Xtr_Norm, label_train)
    pred = knn.predict(Xte_Norm)
    acc = accuracy_score(pred, label_test)
    return acc


def cluster_classify(features, gt_label, n_classes, kmeans=None, max_iter=1000):
    """Performs clustering and evaluates it with bipartitate graph matching."""
    if kmeans is None:
        kmeans = sklearn.cluster.KMeans(
            n_clusters=n_classes,
            max_iter=max_iter,
            random_state=0
        )
    kmeans = kmeans.fit(features)
    pred_label = kmeans.predict(features)
    return bipartite_match(pred_label, gt_label, n_classes)


def bipartite_match(pred, gt, n_classes=None, presence=None):
    """Does maximum biprartite matching between `pred` and `gt`."""

    if n_classes is not None:
        n_gt_labels, n_pred_labels = n_classes, n_classes
    else:
        n_gt_labels = np.unique(gt).shape[0]
        n_pred_labels = np.unique(pred).shape[0]

    cost_matrix = np.zeros([n_gt_labels, n_pred_labels], dtype=np.int32)
    for label in range(n_gt_labels):
        label_idx = (gt == label)
        for new_label in range(n_pred_labels):
            errors = np.equal(pred[label_idx], new_label).astype(np.float32)
            if presence is not None:
                errors *= presence[label_idx]

            num_errors = errors.sum()
            cost_matrix[label, new_label] = -num_errors

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    num_correct = -cost_matrix[row_idx, col_idx].sum()
    acc = float(num_correct) / gt.shape[0]
    return dict(
        assingment=(list(row_idx), list(col_idx)),
        acc=acc,
        num_correct=num_correct
    )
