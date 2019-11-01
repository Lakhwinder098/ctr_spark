# Preprocessing
import pandas as pd
df = pd.read_csv('dataset/parsed_ctr_500000.csv')
# Converting Date and time
df['hour'] = pd.to_datetime(df.hour).dt.hour
# Removing Unwanted Features
colomns_to_drop = ['id','site_id','site_domain','app_id','app_domain','device_id','device_ip','device_model','C1','C14','C15','C16','C17','C18','C19','C20','C21']

df = df.drop(colomns_to_drop,axis=1)
df.to_csv(r'dataset/preprocessed_500000.csv')
