from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.metrics import accuracy_score
import utils

if __name__ == "__main__":
    images = utils.get_all_images()
    labels, features = utils.preprocessing(images, feature_flag="haar", label_flag="expression")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.7
    )

    models = [
        ("SVC", SVC(C=1.5)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, criterion='entropy')),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("GaussianProcess", GaussianProcessClassifier(kernel=kernels.RBF(1.0))),
        ("Decision Tree", DecisionTreeClassifier(ccp_alpha=1.0)),
        ("Xgboost", XGBClassifier(n_estimators=100, learning_rate=1e-3)),
        ("LogisticRegression", LogisticRegression())
    ]

    result = {}

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result[name] = accuracy_score(y_test, y_pred)

    ax = sns.barplot(x=[key for key in result.keys()], y=[item[1] for item in result.items()], palette="deep")
    plt.xlabel("cls name")
    plt.ylabel("% of Acc")
    plt.title("Acc of different cls")
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f"{height: .6%}", (x+width/2, y+height/2), ha='center')
    plt.show()

