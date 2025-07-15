# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Google Ads Campaign Dashboard")

df = pd.read_excel("Google_Ads_Campaign_Analysis_AayushTiwari.xlsx", sheet_name="Campaign Data")
df['CTR'] = df['Clicks'] / df['Impressions']
df['CPC'] = df['Cost (USD)'] / df['Clicks']
df['Conversion Rate'] = df['Conversions'] / df['Clicks']

region = st.selectbox("Select Region", df['Region'].unique())
filtered_df = df[df['Region'] == region]

st.write(f"Showing data for **{region}**:")
st.dataframe(filtered_df)

# Plot CTR by Device
fig, ax = plt.subplots()
sns.barplot(data=filtered_df, x='Device', y='CTR', ax=ax)
ax.set_title("CTR by Device")
st.pyplot(fig)

# Show top campaigns
st.subheader("📈 Top Campaigns by Conversion Rate")
st.dataframe(filtered_df.sort_values(by='Conversion Rate', ascending=False).head(5))