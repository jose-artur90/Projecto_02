import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.4

def main():
    evidence, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=TEST_SIZE)

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)

    y_pred_dtc = dtc.predict(X_test)

    acc_dtc = accuracy_score(y_test, y_pred_dtc)
    conf = confusion_matrix(y_test, y_pred_dtc)
    clf_report = classification_report(y_test, y_pred_dtc)

    print(f"Accuracy Score of Decision Tree is : {acc_dtc}\n")
    print(f"Confusion Matrix : \n{conf}\n")
    print(f"Classification Report : \n{clf_report}\n")
    

def load_data():
    data = pd.read_csv('hotel_booking.csv')
    data.drop_duplicates(inplace=False)
    evidence = data[['lead_time','arrival_date_week_number','arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights']]
    labels = data['is_canceled']
    
    return evidence, labels

if __name__ == "__main__":
    main()
