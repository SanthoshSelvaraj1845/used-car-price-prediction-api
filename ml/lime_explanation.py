import joblib
import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer


# ------------------------------------------------
# 1. Load trained model
# ------------------------------------------------

model = joblib.load(
    "ml/saved_model/model.joblib"
)


# ------------------------------------------------
# 2. Load training dataset
# ------------------------------------------------

data = pd.read_csv(
    "data/used_cars.csv"
)


# ------------------------------------------------
# 3. Features used by the model
# ------------------------------------------------

feature_columns = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


categorical_columns = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


X = data[feature_columns].copy()


# ------------------------------------------------
# 4. Convert categorical columns to integer codes
# ------------------------------------------------

category_mappings = {}

for column in categorical_columns:

    categories = X[column].astype(str).unique()

    mapping = {
        value: index
        for index, value in enumerate(categories)
    }

    category_mappings[column] = mapping

    X[column] = (
        X[column]
        .astype(str)
        .map(mapping)
    )


# Make sure numeric columns are numeric

X["year"] = X["year"].astype(float)
X["km_driven"] = X["km_driven"].astype(float)


# ------------------------------------------------
# 5. Training data for LIME
# ------------------------------------------------

training_data = X.values.astype(float)


# ------------------------------------------------
# 6. Tell LIME which features are categorical
# ------------------------------------------------

categorical_features = [
    feature_columns.index(column)
    for column in categorical_columns
]


categorical_names = {
    feature_columns.index(column):
    list(category_mappings[column].keys())
    for column in categorical_columns
}


# ------------------------------------------------
# 7. Convert LIME numeric data back to original
#    values before sending to our ML pipeline
# ------------------------------------------------

def convert_lime_data(data_array):

    dataframe = pd.DataFrame(
        data_array,
        columns=feature_columns
    )

    for column in categorical_columns:

        mapping = category_mappings[column]

        reverse_mapping = {
            index: value
            for value, index in mapping.items()
        }

        dataframe[column] = (
            dataframe[column]
            .round()
            .astype(int)
            .map(reverse_mapping)
        )

    dataframe["year"] = dataframe["year"].astype(int)

    dataframe["km_driven"] = (
        dataframe["km_driven"].astype(float)
    )

    return dataframe


# ------------------------------------------------
# 8. Prediction function used by LIME
# ------------------------------------------------

def predict_function(data_array):

    dataframe = convert_lime_data(data_array)

    return model.predict(dataframe)


# ------------------------------------------------
# 9. Create LIME explainer
# ------------------------------------------------

explainer = LimeTabularExplainer(
    training_data,
    feature_names=feature_columns,
    categorical_features=categorical_features,
    categorical_names=categorical_names,
    mode="regression",
    discretize_continuous=True,
    random_state=42
)


# ------------------------------------------------
# 10. Function to encode one car
# ------------------------------------------------

def encode_car(car):

    encoded_car = []

    for column in feature_columns:

        value = car[column]

        if column in categorical_columns:

            mapping = category_mappings[column]

            encoded_value = mapping.get(
                str(value),
                0
            )

            encoded_car.append(encoded_value)

        else:

            encoded_car.append(float(value))

    return np.array(
        encoded_car,
        dtype=float
    )


# ------------------------------------------------
# 11. Car 1
# ------------------------------------------------

car_1 = {
    "name": "Maruti Swift VXI",
    "year": 2020,
    "km_driven": 45000,
    "fuel": "Diesel",
    "seller_type": "Dealer",
    "transmission": "Manual",
    "owner": "First Owner"
}


car_1_encoded = encode_car(car_1)


prediction_1 = predict_function(
    np.array([car_1_encoded])
)[0]


explanation_1 = explainer.explain_instance(
    car_1_encoded,
    predict_function,
    num_features=7
)


print("\n===================================")
print("CAR 1")
print("===================================")

print(
    f"Predicted Price: ₹{prediction_1:,.2f}"
)

print("\nLIME Explanation:")

for feature, weight in explanation_1.as_list():

    print(
        f"{feature}: {weight:+.4f}"
    )


explanation_1.save_to_file(
    "ml/saved_model/lime_car_1.html"
)


# ------------------------------------------------
# 12. Car 2
# ------------------------------------------------

car_2 = {
    "name": "Maruti Alto 800",
    "year": 2015,
    "km_driven": 90000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "Second Owner"
}


car_2_encoded = encode_car(car_2)


prediction_2 = predict_function(
    np.array([car_2_encoded])
)[0]


explanation_2 = explainer.explain_instance(
    car_2_encoded,
    predict_function,
    num_features=7
)


print("\n===================================")
print("CAR 2")
print("===================================")

print(
    f"Predicted Price: ₹{prediction_2:,.2f}"
)

print("\nLIME Explanation:")

for feature, weight in explanation_2.as_list():

    print(
        f"{feature}: {weight:+.4f}"
    )


explanation_2.save_to_file(
    "ml/saved_model/lime_car_2.html"
)


print("\n===================================")
print("LIME EXPLANATIONS CREATED")
print("===================================")

print(
    "lime_car_1.html created successfully."
)

print(
    "lime_car_2.html created successfully."
)