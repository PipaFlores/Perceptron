
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.datasets import load_iris


# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
# Data is already shuffled in this case
df = shuffle(df)

print(df.head())

