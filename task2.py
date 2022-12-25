import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.model_selection import train_test_split
import utils

if __name__ == "__main__":
    images = utils.get_all_images()
    labels, features = utils.preprocessing(images, label_flag="expression")
    model = KMeans(n_clusters=4)
    model.fit(features)
    y_pred = model.predict(features)
    features = np.array(features)
    plt.scatter(features[:, 0], features[:, 1], c=y_pred)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.8
    )

    rng = np.random.RandomState(40)
    random_unlabeled_points = rng.random(len(y_train)) < 0.3
    semi_y_train = np.copy(y_train)
    semi_y_train[random_unlabeled_points] = -1
    model = LabelPropagation(n_neighbors=4, gamma=0.5, tol=1e-8, max_iter=10000)
    model.fit(X_train, semi_y_train)
    y_pred = model.predict(X_test)

    cls = GaussianProcessClassifier(kernel=kernels.RBF(1.0))
    cls.fit(X_train, y_train)
    cls_pred = cls.predict(X_test)

    y = [accuracy_score(y_test, cls_pred), accuracy_score(y_test, y_pred)]
    ax = sns.barplot(x=["Gaussian Process", "Label Propagation"], y=y, palette="deep")
    plt.xlabel("model name")
    plt.ylabel("% of Acc")
    plt.title("Acc of different cls")
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f"{height: .6%}", (x+width/2, y+height/2), ha='center')
    plt.show()


