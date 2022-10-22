from sklearn.datasets import dump_svmlight_file, make_classification
import os


data_dir = '/Volumes/Samsung_T5/hal985u/Research/Aspide/Exp-Data'
X, y = make_classification(
    n_samples=600000,
    n_features=650,
)

test = os.path.join(data_dir, "test.txt")
dump_svmlight_file(X, y, f=test, zero_based=False)
