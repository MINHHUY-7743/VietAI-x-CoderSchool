#AG News
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from transformers import pipeline


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
# train_data = train_data.sample(frac=0.5, random_state=42)
# feature_cols = ['Title', 'Description']
# target_col = 'Class Index'

train_data['text'] = train_data['Title'] + " " + train_data['Description']
test_data['text'] = test_data['Title'] + " " + test_data['Description']

x_train = train_data['text']
y_train = train_data['Class Index']
x_test = test_data['text']
y_test = test_data['Class Index']

pipeline = Pipeline([
    ("preprocessor", TfidfVectorizer()),
    ("feature_selector", SelectKBest(chi2, k=500)),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "preprocessor__ngram_range": [(1,1), (1,2)],
    "preprocessor__min_df": [0.01, 0.05],
    "preprocessor__max_df": [0.85, 0.95],

    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": ["balanced", None],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("Tìm kiếm tham số tốt nhất...")
grid_search.fit(x_train, y_train)

print("Best parameters: ", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

best_model = grid_search.best_params_
y_predict = best_model.predict(x_test)

print("\nBáo cáo phân loại trên tập kiểm tra:")
print(classification_report(y_test, y_predict))