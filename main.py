import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

#1. loading the data set from the csv file 
df = pd.read_csv('car_data.csv')
#using multipe feature for predicting the price seperatomg 
#dependent and independent variable fro thed ata set 

features = ['year', 'mileage', 'engine_size','horsepower']
X = df[features]
y = df['price']
 
#trian the multiple linear regression model
model = LinearRegression()

model.fit(X,y)

def interactive_prediction():

    # continuously prompys the suer for used car details 
    # predicts the price and prints the result
    print("=== used car price prediction ===")
    print("enter the details for used car to predict its price")
    print("type exit at any prompt to quit.\n")

while True:
    try:
        input_year =  input("Year (e.g.,2018): ")
        if input_year.lower() == 'exit':
            break
        year = float(input_year)
         
        input_mileage =  input("milage (e.g.,1500): ")
        if input_mileage.lower() == 'exit':
            break
        mileage = float(input_mileage)

        input_engine_size =  input("Year (e.g.,2.0): ")
        if input_engine_size.lower() == 'exit':
            break
        engine_size = float(input_engine_size)

        input_horsepower = input("horsepower (e.g. , 160): ")
        if input_horsepower.lower() == 'exit':
            break
        horsepower = float(input_horsepower)

                 # Form the input array and predict the price
        user_input = np.array([[year, mileage, engine_size, horsepower]])
        predicted_price = model.predict(user_input)[0]
        print(f"\nPredicted used car price: ${predicted_price:,.2f}\n")

    except ValueError:
        print("invalid input please enetr valid numbers\n")
    cont = input("would youlike to predict more? (yes/no)")
    if cont.strip().lower() != 'yes':
        break

def plot_actual_vs_prediction():

#  generates the scatter plot compariing the actual proces and the models predicted
#  prices on the training data
 #predict prices for thr trainng set

 y_pred =  model.predict(X)
 plt.scatter(y, y_pred, color='blue', label='training data')

  #plotting the ideal prediction line 
plt.plot([y.min(), y.max() ], [y.min(), y.max() ], 'r--', label='idela prediction')
plt.xlabel('actual price')
plt.ylabel('predicted price')
plt.title('actual vs predicted prices')
plt.legend()
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.show()
print('plot save as image ')

if __name__ == '__main__':
    interactive_prediction()
    print("Generating Actual vs. Predicted Price Plot...")
    plot_actual_vs_prediction()