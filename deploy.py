import streamlit as st
import pandas as pd
import numpy as np
import requests, re
import pickle
import math 
SEED = int(math.sqrt(201401004 + 191401009))

st.write("""# Simple Movie Prediction App\nThis app predicts the movie revenue!""")
st.sidebar.header('Movie information')

# images and help with url
if st.button('need help with url?', on_click=None):
    img_paths = ["avengers1.png", "avengers2.png", "andreas1.png", "andreas2.png", "i_am_mother1.png", "i_am_mother2.png"]
    import random
    img_no = 2*(random.randint(0, len(img_paths)/2-1))
    
    from PIL import Image
    st.image(Image.open(img_paths[img_no]), caption='andreas', width=750)
    st.image(Image.open(img_paths[img_no+1]), caption='andreas', width=750)
    if st.button('thanks, I learned:)', on_click=None):
        pass        
else:
    pass


# taking input
def user_input_features():
    input_rating_url = st.text_input("IMDb site Ratings page url")
    st.write("\n(all other input other than **Year** gets disabled when url is provided. You should input **Year** always.)")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    if input_rating_url is None or input_rating_url == "":
        input_Year= st.sidebar.selectbox('Year', range(1990, 2023))
        input_Rating = st.sidebar.slider('Rating', min_value=float(0.0), value=float(5.0), max_value=float(10.0), step=0.1)
        input_1 = st.sidebar.number_input('Number of votes as 1', value=3000, format="%i")
        input_2 = st.sidebar.number_input('Number of votes as 2', value=3000, format="%i")
        input_3 = st.sidebar.number_input('Number of votes as 3', value=3000, format="%i")
        input_4 = st.sidebar.number_input('Number of votes as 4', value=3000, format="%i")
        input_5 = st.sidebar.number_input('Number of votes as 5', value=3000, format="%i")
        input_6 = st.sidebar.number_input('Number of votes as 6', value=3000, format="%i")
        input_7 = st.sidebar.number_input('Number of votes as 7', value=3000, format="%i")
        input_8 = st.sidebar.number_input('Number of votes as 8', value=3000, format="%i")
        input_9 = st.sidebar.number_input('Number of votes as 9', value=3000, format="%i")
        input_10 = st.sidebar.number_input('Number of votes as 10', value=3000, format="%i")
        data = {
            'Year' : input_Year,
            'Rating' : input_Rating,
            'Votes' : input_1 + input_2 + input_3 + input_4 + input_5 + input_6 + input_7 + input_8 + input_9 + input_10,
            '1': input_1, '2': input_2, '3': input_3, '4': input_4, '5': input_5, '6': input_6, '7': input_7, '8': input_8, '9': input_9, '10': input_10}
        return pd.DataFrame(data, index=[0])

    else:
        try:
            request_text = requests.get(input_rating_url).text
            star_votes = re.findall("<div class=\"leftAligned\">(.*?)</div>", request_text)
            rating = re.findall("span class=\"ipl-rating-star__rating\">(.*?)</span>", request_text)
            stars = [int(vote.replace(",","")) for vote in star_votes[1:11]]

            input_Year= st.sidebar.selectbox('Year', range(1990, 2023))
            [input_10, input_9, input_8, input_7, input_6, input_5, input_4, input_3, input_2, input_1] = stars
            input_Rating = float(rating[0])

            data = {
                'Year' : input_Year,
                'Rating' : input_Rating,
                'Votes' : input_1 + input_2 + input_3 + input_4 + input_5 + input_6 + input_7 + input_8 + input_9 + input_10,
                '1': input_1, '2': input_2, '3': input_3, '4': input_4, '5': input_5, '6': input_6, '7': input_7, '8': input_8, '9': input_9, '10': input_10}

            return pd.DataFrame(data, index=[0])
        except:
            st.write("Pwease input infowmations by hand")
            st.write("🥺👉👈")
            raise Exception("*Sowwy, unable to get data fwom pwovided page uwl*😔")


# scaling input
def scale_raw_input_for_NN(raw_input):
    data = raw_input.copy()
    import numpy as np
    year_revenue_dict = {1990: 0.7658344741262136, 1991: 0.6158904723529411, 1992: 0.5810284048958334, 1993: 0.5947457455973276, 1994: 0.5941740310769231, 1995: 0.6217016949917985, 1996: 0.6502561518881119, 1997: 0.6338443119205298, 1998: 0.8677550960544218, 1999: 0.6733860998742138, 2000: 0.7771750025433526, 2001: 0.7466757578888888, 2002: 0.708450799753397, 2003: 0.7759085865470852, 2004: 0.8238424626760563, 2005: 0.782264222, 2006: 0.7498834795081967, 2007: 0.6502655863192183, 2008: 0.7055149379672131, 2009: 0.8754953023278688, 2010: 0.6809290582777777, 2011: 0.9059954949253732, 2012: 0.811775069737705, 2013: 0.8438092751023868, 2014: 0.8119720837606839, 2015: 0.8807708197784809, 2016: 0.82369238359375, 2017: 1.0752321504301077, 2018: 0.8333133838255032, 2019: 0.9456890671153846, 2020: 0.3045369520779221, 2021: 0.9344049581962025, 2022: 0.9695750887288135}
    data['Year'] = data['Year'].map(year_revenue_dict)
    data["Rating"] = data["Rating"]/10

    data["1%"] = data["1"] / data["Votes"]
    data["2%"] = data["2"] / data["Votes"]
    data["3%"] = data["3"] / data["Votes"]
    data["4%"] = data["4"] / data["Votes"]
    data["5%"] = data["5"] / data["Votes"]
    data["6%"] = data["6"] / data["Votes"]
    data["7%"] = data["7"] / data["Votes"]
    data["8%"] = data["8"] / data["Votes"]
    data["9%"] = data["9"] / data["Votes"]
    data["10%"] = data["10"] / data["Votes"]

    data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']] = np.log2(data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])

    min_max_scaler = pickle.load(open('MinMaxScaler.pickle', 'rb'))
    data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]= min_max_scaler.transform(data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])

    return data


# input display
raw_input = user_input_features()
st.subheader('Inputted Movie informations')
st.write(raw_input)
scaled_input_for_NN = scale_raw_input_for_NN(raw_input)


# predictiong
from keras.models import load_model
NN_paths = ["kerasNN14_wo_log.h5", "kerasNN16_wo_log.h5", "kerasNN23_wo_log.h5", "kerasNN24_wo_log.h5", "kerasNNtuner_wo_log-001-24.h5", "kerasNNtuner2_wo_log-001-24.h5"]
sum_prediction = predictors = 0
for NN_h5 in NN_paths:
    try:
        NN = load_model(NN_h5)
        sum_prediction = sum_prediction + np.log2(NN.predict(scaled_input_for_NN)*100581)
        predictors = predictors + 1
    except Exception as e:
        print(e)
        

# printing
avg_prediction = sum_prediction / predictors

lower_bound = 2**(avg_prediction[0][0]-0.75)
revenue_pred = 2**(avg_prediction[0][0])
upper_bound = 2**(avg_prediction[0][0]+0.75)

lower_desired_representation = "{:,.2f}".format(lower_bound)
desired_representation = "{:,.2f}".format(revenue_pred)
upper_desired_representation = "{:,.2f}".format(upper_bound)

st.subheader('Prediction')
st.write("$", desired_representation)
st.write("expected to be between", "$", lower_desired_representation, "-", upper_desired_representation,".")
