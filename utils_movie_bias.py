import ast
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def str_to_list(s):
    return ast.literal_eval(s)

def filter_column_types(df, column, minRowsFeaturesPresent):
    # Remove the least frequent subtypes for Cast, Genre, and Director
    feature_counts = df[column].explode().value_counts()
    top_feature_subtypes = set(feature_counts[feature_counts >= minRowsFeaturesPresent].index)
    # Filter the Cast column based on the remaining values
    filtered_feature = df[column].apply(lambda x: [i for i in x if i in top_feature_subtypes])
    # Apply one-hot encoding on feature
    mlb = MultiLabelBinarizer()
    features_encoded = pd.DataFrame(mlb.fit_transform(filtered_feature), columns=column+'_'+mlb.classes_)
    return features_encoded


