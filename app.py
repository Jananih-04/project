from flask import Flask, request, render_template, redirect, url_for
from joblib import load
import pandas as pd
import numpy as np
import csv

app = Flask(__name__)

# Load your trained models with updated paths
rf_model = load(r"C:\Users\Sowmya H\Downloads\project\project\randomforest.joblib")
gbm_model = load(r"C:\Users\Sowmya H\Downloads\project\project\Gradiant.joblib")
ensemble_model = load(r"C:\Users\Sowmya H\Downloads\project\project\ensemble.joblib")

# Load the symptoms data to get the list of symptoms
df_train = pd.read_csv(r"C:\Users\Sowmya H\Downloads\project\project\Training.csv")
symptoms_list = df_train.columns.tolist()
symptoms_list.remove("prognosis")

# Load symptom severity and precaution data
symptom_severity_df = pd.read_csv(r"C:\Users\Sowmya H\Downloads\project\project\symptom_severity.csv")
disease_precaution_df = pd.read_csv(r"C:\Users\Sowmya H\Downloads\project\project\disease_precaution.csv")


disease_description= pd.read_csv(r"C:\Users\Sowmya H\Downloads\project\project\disease_description.csv")
# Create dictionaries for mapping symptoms to severity and diseases to precautions
symptom_severity_dict = dict(zip(symptom_severity_df['Symptom'], symptom_severity_df['Symptom_severity']))
disease_description_dict=dict(zip(disease_description['Disease'],disease_description['Symptom_Description']))
disease_precaution_dict = {}
for _, row in disease_precaution_df.iterrows():
    disease_precaution_dict[row['Disease']] = row[['Symptom_precaution_0', 'Symptom_precaution_1', 'Symptom_precaution_2']].tolist()

USER_CREDENTIALS = {'username': 'admin', 'password': 'password'}

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get the username and password from the form
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the credentials are correct
        if username == USER_CREDENTIALS['username'] and password == USER_CREDENTIALS['password']:
            # Redirect to the index page after successful login
            return redirect(url_for('index.html'))
        else:
            # If credentials are incorrect, render the login page again with an error message
            return render_template('login.html', error='Invalid username or password')

    # Render the login page for GET requests
    return render_template('login.html', error=None)

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_symptoms = []
        # Retrieve selected symptoms from the form
        for i in range(1, 4):
            symptom = request.form.get(f'symptom{i}')
            if symptom:
                input_symptoms.append(symptom)

        if not input_symptoms:
            return render_template('index.html', error="Please select at least one symptom.", symptoms=symptoms_list)

        # Create an empty DataFrame with all symptoms as columns
        input_data = pd.DataFrame(np.zeros((1, len(symptoms_list))), columns=symptoms_list)

        # Set symptom values to 1 where applicable
        input_data[input_symptoms] = 1

        # Make predictions using ensemble model
        predicted_disease = ensemble_model.predict(input_data)[0]
        
        # Get severity for input symptoms
        severity = get_severity(input_symptoms)
        
        # Get precautions for predicted disease
        precautions = get_precautions(predicted_disease)

        description = get_description(predicted_disease)
        site=get_locations_for_disease(r'C:\Users\USER\Downloads\project\statessss.csv',predicted_disease)
        
        return render_template('result.html', prediction=predicted_disease, severity=severity, precautions=precautions,descriptions=description,location = site)

def get_severity(symptoms):
    # Calculate severity based on average of symptom severities
    avg_severity = np.mean([symptom_severity_dict.get(symptom, 0) for symptom in symptoms])
    return avg_severity

def get_precautions(disease):
    return disease_precaution_dict.get(disease, [])

def get_description(disease):
    return disease_description_dict[disease]

def determine_phase(age, gender, medications, severity):
    phase = 'Phase not determined'
    description = ''

    if age == '18-35' and gender == 'male' and medications == 'no' and severity == 'mild':
        phase = 'Phase 0'
        description = ' In Phase 0 trials, a small number of healthy volunteers are usually recruited. These individuals may include adults who meet specific health criteria and are willing to participate in the study to help researchers understand how the investigational drug behaves in the human body. These trials are primarily focused on pharmacokinetics and may involve limited exposure to the drug'
    elif age == '25-45' and gender == 'female' and medications == 'no' and severity == 'moderate':
        phase = 'Phase I'
        description = 'Phase I trials primarily involve healthy volunteers, although in some cases, individuals with the condition being studied may also participate. Healthy volunteers are carefully screened to ensure they meet the study eligibility criteria and do not have pre-existing health conditions that could interfere with the trial results'
    elif age == '30-65' and gender == 'male' and medications == 'yes' and severity == 'moderate':
        phase = 'Phase II'
        description = ' Phase II trials typically include participants who have the specific condition or disease that the drug is intended to treat. These individuals are selected based on specific inclusion and exclusion criteria related to their medical history, disease severity, and other factors.'
    elif age == '65+' and gender == 'female' and medications == 'yes' and severity == 'severe':
        phase = 'Phase III'
        description = 'Phase III trials involve a larger and more diverse group of participants, including individuals with the target condition or disease who may have varying degrees of severity. These trials often recruit participants from multiple clinical sites to ensure a representative sample of the intended patient population'
    
    return phase, description
@app.route('/phases', methods=['GET', 'POST'])
def phases():
    if request.method == 'POST':
        age = request.form['age']
        gender = request.form['gender']
        medications = request.form['medications']
        severity = request.form['severity']

        phase, description = determine_phase(age, gender, medications, severity)

        return render_template('phases.html', phase=phase, description=description)
    else:
        return render_template('phases.html', phase=None, description=None)

def get_locations_for_disease(csv_file, disease):
    locations = ""
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Disease'] == disease:
                
                locations = row['Siteselected']
                break
    return locations   
    
if __name__ == '__main__':
    app.run(debug=True)
