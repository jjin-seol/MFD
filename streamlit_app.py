import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

# Title
st.title("MOTOR FAULT DIAGNOSIS")

st.subheader("SELECT MODEL")

# Model select & dir setting
model = st.selectbox(
    'Choice train model',
    ('LSTM','LSTM with Attention','GRU', 'GRU with Attention','CNN','CNN with Attention',
    'LSTM,GRU,CNN+Ensemble','LSTM,GRU,CNN+Attention','Attention(LSTM,GRU,CNN)+Ensemble','Attention(LSTM,GRU,CNN)+Attention','X'), 
)
if model == 'LSTM':
    st.markdown('**Choice Model** : :blue[LSTM]')
    model_dir = "lstmB"
if model == 'GRU':
    st.markdown('**Choice Model** : :blue[GRU]' )
    model_dir = "gruB"
if model == 'CNN':
    st.markdown('**Choice Model** : :blue[CNN]' )
    model_dir = "cnnB"

if model == 'LSTM with Attention':
    st.markdown('**Choice Model** : :blue[LSTM] with :red[Attention]')
    model_dir = "lstmA"
if model == 'GRU with Attention':
    st.markdown('**Choice Model** : :blue[GRU] with :red[Attention]')
    model_dir = "gruA"
if model == 'CNN with Attention':
    st.markdown('**Choice Model** : :blue[CNN] with :red[Attention]')
    model_dir = "cnnA"

if model == 'Attention(LSTM,GRU,CNN)+Aesemble':
    st.markdown('**Choice Model** : :red[Attention(LSTM,GRU,CNN)] + :blue[Ensemble]')
    model_dir = "comAB"
if model == 'LSTM,GRU,CNN+Aesemble':
    st.markdown('**Choice Model** : :blue[LSTM,GRU,CNN] + :blue[Ensemble]')
    model_dir = "comBB"
if model == 'LSTM,GRU,CNN+Attention':
    st.markdown('**Choice Model** : :blue[LSTM,GRU,CNN] + :red[Attention]')
    model_dir = "comBA"
if model == 'Attention(LSTM,GRU,CNN)+Attention':
    st.markdown('**Choice Model** : :red[Attention(LSTM,GRU,CNN)] + :red[Attention]')
    model_dir = "comAA"

# Test data upload
st.subheader("DATA UPLOAD")
test_data = st.file_uploader('Test data Upload (.npy)', type="npy")

# Test data info
st.markdown("**test data size = n x 300 x 4**")
st.caption("1. seq_len is 300")
st.caption("2. number of feature is 4 (Stator Voltage, Stator Current, Rotor Current, Motor Speed)")
data = {
    'Fault': ['Variation of Rotor Excitation Current ', 'Rotor Voltage Excitation disconnection', 'One Phase-to-neutral Short Circuit', 'Two Phase Short Circuit','Open phase','No Fault'],
    'Class': [0,1,2,3,4,5],
}
df = pd.DataFrame(data)
st.subheader("CLASS DEFINITION")
st.write(df)

if model_dir is not None and test_data is not None:
    
    # Model load
    model = tf.keras.models.load_model(model_dir)
    
    # Test data load
    test_array = np.load(test_data)

    # predict
    y_pred = model.predict(test_array)

    # output
    st.subheader("PREDICTION CLASS")
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = y_pred.T
    st.write(y_pred)

    st.subheader("EVALUATION")

    # true data upload for evaluation
    ref_data = st.file_uploader("Label data Upload (.npy)", type="npy")
    if ref_data is not None:
        y_test = np.load(ref_data)
        y_test = np.argmax(y_test, axis=1)
        # st.write(y_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
        
        col1, col2,  _ = st.columns(3)
        col1.metric(label="Accuracy", value='%.3f' %accuracy)
        col2.metric(label="F1-Score", value='%.3f' %fscore[0])

        col3, col4, col5 = st.columns(3)
        col3.metric(label="Precision", value='%.3f' %precision[0])
        col4.metric(label="Recall", value='%.3f' %recall[0])
        col5.metric(label="Specificity", value='%.3f' %recall[1])
        
        st.write('confusion_matrix')
        fig = plt.figure(figsize=(7,5))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot(fig)
