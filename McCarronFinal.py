from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

app = Flask(__name__)

#Bring in gradient boosting model with pickle
with open("GradBoost_model.pkl", "rb") as f:
    GradBoost_model = pickle.load(f)

#function for calculating profit
def loan(probability, LoanAmount, LoanTerm, InterestRate):
    profit = probability*LoanAmount*LoanTerm/12*InterestRate/100 - (1-probability)*LoanAmount
    if profit > 0:
        status = 'Approved'
    else:
        status = 'Denied'
    return status

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/resume')
def resume_page():
    return render_template('resume.html')


@app.route('/links')
def projects_page():
    return render_template('projects.html')


@app.route('/loandefault', methods=['POST', 'GET'])
def loandefault_page():
    if request.method == 'POST':
        info_submit = pd.DataFrame(request.form.to_dict(), index=[0])

        non_cat = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                   'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

        cat = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose',
               'HasCoSigner']

        #split between categorical data and numeric
        non_cat_dat = info_submit[non_cat]
        cat_dat = info_submit[cat]

        #One hot encode data
        encoder = OneHotEncoder()
        encoder.fit(cat_dat)
        cat_dat_encoded = encoder.transform(cat_dat)
        cat_dat_encoded = pd.DataFrame(cat_dat_encoded.toarray(), columns=encoder.get_feature_names_out())

        # recombine
        submit_data_OHE = non_cat_dat.merge(cat_dat_encoded, left_index=True, right_index=True)

        #added so that OHE data has necessary variables for Grad Boosting model
        empty_df = pd.DataFrame(
            {'Age': '0', 'Income': '0', 'LoanAmount': '0', 'CreditScore': '0', 'MonthsEmployed': '0', 'NumCreditLines': '0',
             'InterestRate': '0', 'LoanTerm': '0', 'DTIRatio': '0',
             "Education_Bachelor's": '0', 'Education_High School': '0', "Education_Master's": '0', 'Education_PhD': '0',
             'EmploymentType_Full-time': '0',
             'EmploymentType_Part-time': '0', 'EmploymentType_Self-employed': '0', 'EmploymentType_Unemployed': '0',
             'MaritalStatus_Divorced': '0',
             'MaritalStatus_Married': '0', 'MaritalStatus_Single': '0', 'HasMortgage_No': '0', 'HasMortgage_Yes': '0',
             'HasDependents_No': '0', 'HasDependents_Yes': '0',
             'LoanPurpose_Auto': '0', 'LoanPurpose_Business': '0', 'LoanPurpose_Education': '0', 'LoanPurpose_Home': '0',
             'LoanPurpose_Other': '0', 'HasCoSigner_No': '0', 'HasCoSigner_Yes': '0'}, index=[0])

        comb_data = pd.concat([empty_df, submit_data_OHE], ignore_index=True)

        #complete data
        full_data = pd.DataFrame(comb_data.iloc[1]).transpose().fillna(0)
        LoanAmount = float(full_data['LoanAmount'].iloc[0])
        LoanTerm = float(full_data['LoanTerm'].iloc[0])
        InterestRate = float(full_data['InterestRate'].iloc[0])

        #probability predicted by gradient boosting model
        probability = GradBoost_model.predict_proba(full_data)[0][0]
        prediction = loan(probability, LoanAmount, LoanTerm, InterestRate)
        return render_template('loandefault.html', prediction=prediction)
    else:
        prediction = 'Not submitted'
        return render_template('loandefault.html', prediction=prediction)
