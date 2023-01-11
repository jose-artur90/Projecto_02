import csv 
import sys
import pandas as pd
import numpy as numpy

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():

    if len(sys.argv) != 2:
        sys.exit("Usar : python hotel.py data")

        evidence, labels = load_data(sys.argv[1])
        X_train, X_test, y_train, y_test = train_test_split(
            evidence, labels, test_size=TEST_SIZE
        )

        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        sensitivity = specificity = evaluate(y_test , predictions)

        print(f"Correto:{(y_test == predictions).sum()}")
        print(f"Incorreto:{(y_test != predictions).sum()}")
        print(f"Taxa Verdadeiro Positivo:{100*sensitivity:.2f}%")
        print(f"Taxa Verdadeiro Negativo:{100*specificity:.2f}%")

def load_data(filename):

    data = pd.read_csv('hotel_booking.csv', header=0)

    d={
        'January':0,
        'Febuary':1,
        'March':2,
        'April':3,
        'May':4,
        'June':5,
        'July':6,
        'August':7,
        'September':8,
        'October':9,
        'November':10,
        'December':11
    }

    data.arrival_date_month = data.arrival_date_month.map(d)

    data.is_repeated_guest = data.is_repeated_guest.map(lambda x : 1 if x == 'Transient' else 0)

    data.stays_in_weekend_nights = data.stays_in_weekend_nights.map(lambda x : 1 if x == 1 else 0)

    data.is_canceled = data.is_canceled.map(lambda x : 1 if x==1 else 0)
    
    data.reservation_status = data.reservation_status(lambda x : 1 if x == 'Canceled' else 0)           