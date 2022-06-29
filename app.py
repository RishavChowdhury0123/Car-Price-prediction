import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import streamlit as st

st.set_page_config(page_title='Car price predictor', layout='wide')


@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def load_data():
    path= 'df.pkl'
    with open(path, 'rb') as ref:
        df= pickle.load(ref)
    path= 'model.pkl'
    with open(path, 'rb') as ref:
        pipe= pickle.load(ref)
    
    return df, pipe

# To transform numbers to abbreviated format
def format_numbers(number, pos=None, fmt= '.0f'):
    fmt= '%'+fmt
    thousands, millions, billions= 1_000, 1_000_000, 1_000_000_000
    if number/billions >=1:
        return (fmt+'B') %(number/billions)
    elif number/millions >=1:
        return (fmt+'M') %(number/millions)
    elif number/thousands >=1:
        return (fmt+'K') %(number/thousands)
    else:
        return fmt %(number)

# Function for encoding multiple features
class CustomEncoder:

    def __init__(self, columns):
        self.columns= columns
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X):
        from sklearn.preprocessing import LabelEncoder
        out= X.copy()
        if self.columns is not None:
            out[self.columns]= out[self.columns].apply(lambda x: LabelEncoder().fit_transform(x))
        else:
            out= out.apply(lambda x: LabelEncoder().fit_transform(x))
        return out
    
    def fit_transform(self, X, y=None):
        out= X.copy()
        return self.fit(out).transform(out)

def main():
    st.title('Used Car Price predictor')
    cols= st.columns(4)
    brand= cols[0].selectbox('brand', brand_models.keys(), key='brand')
    model= cols[1].selectbox('Model',brand_models[st.session_state.brand], key='model')
    values= model_year.get((brand,model))
    year= cols[2].number_input('Year', min_value= values[0], max_value= values[1], step=1, value= values[0], key='year')
    fueltype= cols[3].radio('Fuel Type', ['Petrol','Diesel'], key='fueltype')
    
    cols= st.columns(4)
    mileage= cols[0].number_input('Distance travelled', min_value= int(df.mileage.min()),max_value= 85000, step=1000, key='mileage')
    trans= cols[1].selectbox('Transmission', model_trans.get((brand, model)), key='trans')
    values= model_mpg.get((brand,model))
    mpg= cols[2].number_input('MPG',min_value= values[0], max_value=values[1], value=np.mean(values), key='mpg')
    values= model_engine.get((brand,model))
    enginesize= cols[3].number_input('Engine size',min_value= values[0], max_value=values[1], value= np.mean(values), key='enginesize')

    vals= [brand, model, year, trans, mileage, fueltype, mpg, enginesize]

    btn= st.button('Predict price', key='button')
    X= df.copy()
    if btn:
        X.drop(['carID','tax'],axis=1, inplace=True)
        X= X.append(dict(zip(X.columns, vals)), ignore_index=True)
        
        X.mileage= X.mileage**0.5

        X= pd.get_dummies(X, columns= ['brand','transmission','fuelType'], drop_first=True)
        X= X.drop(['brand_TOYOTA', 'transmission_Semi-Auto'], axis=1)
        X= CustomEncoder(X.select_dtypes('O').columns).fit_transform(X)

        pred= pipe.predict(X.iloc[-2:,:])[-1]
        pred= format_numbers(pred**2, fmt='.1f')
        st.markdown('Resale value would be approximately around  $%s.'%pred)

df, pipe= load_data()
brand_models= dict(df.groupby('brand').model.unique())
model_trans= dict(df.groupby(['brand','model']).transmission.unique())
model_year= dict(df.groupby(['brand','model']).year.apply(lambda x: [x.min(), x.max()]))
model_mpg= dict(df.groupby(['brand','model']).mpg.apply(lambda x: [x.min(), x.max()]))
model_engine= dict(df.groupby(['brand','model']).engineSize.apply(lambda x: [x.min(), x.max()]))

if __name__=='__main__':
    main()
