from flask import Flask,render_template,request
import pickle
import pandas as pd

application=Flask(__name__)
@application.route('/')
def homepage():
    return render_template('index.html')

@application.route('/submit',methods=['POST'])
def result():
    lblenc=pickle.load(open('labelencoder.pickle','rb'))
    xgb_mod=pickle.load(open('XGBoostIncome.pickle','rb'))
    age=request.form['age']
    population=request.form['fnlwgt']
    education=request.form['edu']
    capital_gain=request.form['cap_g']
    capital_loss=request.form['cap_l']
    hour_per_week=request.form['hrpw']
    workclass=request.form['wrkcls']
    workclass_local_gvt=request.form['wrkcls_loc']
    workclass_never=request.form['wrk_never']
    workclass_private=request.form['wrk_pvt']
    workclass_self=request.form['wrk_self']
    workclass_selfnot=request.form['wrk_selfnot']
    workclass_state_gvt=request.form['wrk_state']
    work_wo_pay=request.form['wrk_w/o_pay']
    ed10th=request.form['ed10th']
    ed11th=request.form['ed11th']
    result= xgb_mod.predict(pd.DataFrame([age,population,education,capital_gain,capital_loss,hour_per_week,workclass,
    workclass_local_gvt,workclass_never,workclass_private,workclass_self,workclass_selfnot,workclass_state_gvt,
    work_wo_pay,ed10th,ed11th]).values.T)
    final=lblenc.inverse_transform(result)
    return 'Predicted Salary of an Individual as per entered data is:'+str(final)[1:-1]
if __name__=='__main__':
    application.run(debug=True)
