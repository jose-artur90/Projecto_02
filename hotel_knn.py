import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():

    evidence, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=TEST_SIZE)

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test , predictions)

    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100*sensitivity:.2f}%")
    print(f"True Negative Rate: {100*specificity:.2f}%")
    

def load_data():
    data = pd.read_csv('hotel_booking.csv')
    data.drop_duplicates(inplace=False)
    evidence = data[['lead_time','arrival_date_week_number','arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights']]
    labels = data['is_canceled']
    return evidence, labels

def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    sensitivity = np.count_nonzero(np.logical_and(labels == 1, predictions == 1)) / np.count_nonzero(labels)
    specificity = np.count_nonzero(np.logical_and(labels == 0, predictions == 0)) / (len(labels) - np.count_nonzero(labels))
    return sensitivity, specificity

if __name__ == "__main__":

    main()
