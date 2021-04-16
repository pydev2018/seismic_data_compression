import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title=" Skin Segmentation",
    page_icon=":shark", 
    layout="wide"# EP: how did they find a symbol?
    
    
)

st.title('Seismic data compression using Singular Value decomposition')

st.image("svd.JPG")
st.image("svd_2.JPG")
st.image("svd_3.JPG")

st.markdown(" ### The data used for this experiment is taken from Mobil AVO viking graben line 12")

num_k = st.slider(
    "Set the K-rank for SVD compression", 0, 120, 50, 1
)




@st.cache
def load_data():
    data = np.load('one_shot.npy')
    data = np.array(data , dtype='float32')
    vm = np.percentile(data,99)
    return data , vm 
    
data , vm = load_data()

df_a = pd.DataFrame(data)

def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

fig_data  = go.Heatmap(df_to_plotly(df_a) ,  colorscale='greys', zmin=-vm, zmax=vm, zsmooth='best', showscale=False)
fig = {'data': [fig_data], 
        'layout' : go.Layout(xaxis_nticks=20, yaxis_nticks=20)} 
fig['layout']['yaxis']['autorange'] = "reversed"
fig['layout']['width'] = 500
fig['layout']['height'] = 800
fig['layout']['title'] = "Original Gather, size {} bytes".format(data.nbytes)


@st.cache
def getcompressed_svd(data, num_k):
    row , col = data.shape
    U_s, d_s, V_s = np.linalg.svd(data, full_matrices=True)
    k = num_k
    U_s_k = U_s[:, 0:k]
    V_s_k = V_s[0:k, :]
    d_s_k = d_s[0:k]
    compressed_bytes = sum([matrix.nbytes for matrix in [U_s_k, d_s_k, V_s_k]])
    image_approx = np.dot(U_s_k, np.dot(np.diag(d_s_k), V_s_k))
    image_reconstructed = np.zeros((row, col))
    image_reconstructed[:, :] = image_approx
    
    return image_reconstructed , compressed_bytes


reconst_image , comp_size = getcompressed_svd(data, num_k) 
ratio = comp_size / data.nbytes
ratio = round(ratio, 1)

fig_data_svd = go.Heatmap(df_to_plotly(pd.DataFrame(reconst_image)) ,  colorscale='greys', zmin=-vm, zmax=vm, zsmooth='best', showscale=False)
fig_svd = {'data': [fig_data_svd], 
        'layout' : go.Layout(xaxis_nticks=20, yaxis_nticks=20)} 
fig_svd['layout']['yaxis']['autorange'] = "reversed"
fig_svd['layout']['width'] = 500
fig_svd['layout']['height'] = 800
fig_svd['layout']['title'] = "Compressed Gather,size {} bytes ,ratio {}".format(comp_size, ratio)



@st.cache
def get_diff(data, reconst_image):
    diff = data - reconst_image
    pd_diff = pd.DataFrame(diff)
    return pd_diff

pd_diff = get_diff(data, reconst_image)

fig_data_svd_diff = go.Heatmap(df_to_plotly(pd.DataFrame(pd_diff)) ,  colorscale='greys', zmin=-vm, zmax=vm, zsmooth='best', showscale=False)
fig_svd_diff = {'data': [fig_data_svd_diff], 
        'layout' : go.Layout(xaxis_nticks=20, yaxis_nticks=20)} 
fig_svd_diff['layout']['yaxis']['autorange'] = "reversed"
fig_svd_diff['layout']['width'] = 500
fig_svd_diff['layout']['height'] = 800
fig_svd_diff['layout']['title'] = "Difference"




col1, col2, col3 = st.beta_columns(3)
col1.plotly_chart(fig)
col2.plotly_chart(fig_svd)
col3.plotly_chart(fig_svd_diff)





#col2.ploty_chart(fig_svd)
#st.plotly_chart(fig)