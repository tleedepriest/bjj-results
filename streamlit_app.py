import json
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
import numpy as np
import streamlit as st
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


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

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
st.dataframe(filter_dataframe(data))


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
