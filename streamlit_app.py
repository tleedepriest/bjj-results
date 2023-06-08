import json
import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/devstorage.read_only']
SERVICE_ACCOUNT_FILE = '/media/sf_VM_shared/bjj-lineage-streamlit-access-key.json'
# when deploying, copy contents of key file to secrets.toml file and 
# st.secrets can be accessed as a dictionary that you can pass here.
# change _file to _info

CREDENTIAL = service_account.Credentials.from_service_account_info(
        st.secrets["cloud_secrets"], scopes=SCOPES)

FILENAME = "test.csv"

# extract cols from timestamp col...
DATE_COL = "date"
DATETIME_COL = "file_path"
YEAR_COL = "year"
BUCKET_NAME = "bjj-lineage-ibjjf-events-results-all-parsed-json"
STORAGE_CLIENT = storage.Client(credentials=CREDENTIAL)


@st.cache_data
def load_data(**kwargs):
    rows = []
    blobs = STORAGE_CLIENT.list_blobs(BUCKET_NAME)
    for blob in blobs:
        with blob.open('r') as fh:
            for line in fh:
                row = json.loads(line)
                rows.append(row)
    data = pd.DataFrame.from_records(rows, **kwargs)
    data[YEAR_COL] = data[DATETIME_COL].str.extract(r'(\d{4})')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

st.title("IBJJF Results")
data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Done! (using st.cache_data)")



metric_counts = data.groupby(['file_path', YEAR_COL]).size().reset_index(name='counts').groupby(YEAR_COL).size().reset_index(name="counts")
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(metric_counts)
st.subheader('Number of IBJJF Events Per Year')
#hist_values = np.histogram(metric_counts, bins=len(metric_counts), range=(data[YEAR_COL].min(), data[YEAR_COL].max()))
st.bar_chart(metric_counts, y="counts", x="year")

# Some number in the range 2012-2022
#year_to_filter = st.slider('year', 2012, 2022, 2013)
#filtered_data = data[data[YEAR_COL] == str(year_to_filter)]

#st.subheader('Map of all pickups at %s:00' % hour_to_filter)
#st.map(filtered_data)