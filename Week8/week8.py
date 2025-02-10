import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def filter_location(location):
    result = location.split(",")
    if len(result) > 1:
        return result[1][1:]
    else:
        return location

data = pd.read_excel("job_dataset.ods", engine="odf", dtype="str")
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)

target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

small_classes = ["director_business_unit_leader", "managing_director_small_medium_company", "specialist"]
y = y.replace(small_classes, "other_directors")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2801, stratify=y)

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.99), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry"),
])

model_general = Pipeline(steps=[
    ("pre_processor", preprocessor),
    ("feature_selector", SelectKBest(chi2, k=500)),
    ("classifier", RandomForestClassifier(random_state=100, class_weight="balanced"))
])

model_general.fit(x_train, y_train)
y_predict_general = model_general.predict(x_test)

print("Classification report for general model:")
print(classification_report(y_test, y_predict_general))

x_other = x[y == "other_directors"]
y_other = data[target][y == "other_directors"]

x_train_other, x_test_other, y_train_other, y_test_other = train_test_split(
    x_other, y_other, test_size=0.2, random_state=2801, stratify=y_other
)

model_sub = Pipeline(steps=[
    ("pre_processor", preprocessor),
    ("feature_selector", SelectKBest(chi2, k=100)),
    # ("fearture_selector", SelectPercentile(chi2, percentile=5)),#Selectpercentile
    ("classifier", RandomForestClassifier(random_state=100, class_weight="balanced"))
])

model_sub.fit(x_train_other, y_train_other)
y_predict_sub = model_sub.predict(x_test_other)

print("Classification report for sub model:")
print(classification_report(y_test_other, y_predict_sub))

# Classification report for general model:
#                                       precision    recall  f1-score   support
#
#                       bereichsleiter       0.83      0.20      0.33       192
#                  manager_team_leader       0.66      0.72      0.69       534
#                      other_directors       1.00      0.29      0.44        21
# senior_specialist_or_project_manager       0.82      0.93      0.87       868
#
#                             accuracy                           0.76      1615
#                            macro avg       0.83      0.53      0.58      1615
#                         weighted avg       0.77      0.76      0.74      1615
#
# Classification report for sub model:
#                                         precision    recall  f1-score   support
#
#          director_business_unit_leader       0.74      1.00      0.85        14
# managing_director_small_medium_company       0.00      0.00      0.00         1
#                             specialist       1.00      0.33      0.50         6
#
#                               accuracy                           0.76        21
#                              macro avg       0.58      0.44      0.45        21
#                           weighted avg       0.78      0.76      0.71        21