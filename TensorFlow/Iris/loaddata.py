import pandas as pd
import tensorflow as tf

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def read_data():
  data = pd.read_csv('sampledata.csv', names=CSV_COLUMN_NAMES, header=0)
  return data
