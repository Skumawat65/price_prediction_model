import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor 

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())  
])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")) 
]) 

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)   
]) 

    return full_pipeline  

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("bengaluru_house_prices.csv")
    housing['price_cat']= pd.cut(housing["price"],
                                   bins=[0, 50, 72, 100, 150, np.inf],
                                   labels=[1,2,3,4,5]) 
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, _ in split.split(housing, housing['price_cat']):
    housing.loc[train_index].drop("price_cat", axis=1).to_csv("input.csv", index=False) 
    housing = housing.loc[train_index].drop("price_cat", axis=1) 

    housing_labels = housing["price"].copy()
    housing_features = housing.drop("price", axis=1)
    num_attribs = housing_features.select_dtypes(include=[np.number]).columns.tolist() 
    cat_attribs =housing_features.select_dtypes(exclude=[np.number]).columns.tolist()  

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels) 

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved.") 

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["price"]= predictions 

    input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")  