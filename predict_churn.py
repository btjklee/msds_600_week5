import pandas as pd
from pycaret.classification import predict_model, load_model

def predict_churn(input_data):
    # Load the saved model
    loaded_model = load_model('pycaret_model')
 
    input_data['TotalCharges_to_tenure'] = input_data['TotalCharges'] / input_data['tenure']
    
    # Make predictions on the input data
    predictions = predict_model(loaded_model, data=input_data)

    return predictions['prediction_score']

if __name__ == "__main__":
    new_data = pd.read_csv('new_churn_data.csv') 
    true_values = [1, 0, 0, 1, 0]  
    churn_probabilities = predict_churn(new_data)
    print("True Values:", true_values)
    print("Churn Probabilities:")
    print(churn_probabilities)