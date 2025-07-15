# app.py (Flask version)
from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    df = pd.read_excel("Google_Ads_Campaign_Analysis_AayushTiwari.xlsx", sheet_name="Campaign Data")
    df['CTR'] = df['Clicks'] / df['Impressions']
    df['CPC'] = df['Cost (USD)'] / df['Clicks']
    df['Conversion Rate'] = df['Conversions'] / df['Clicks']
    top_campaigns = df.sort_values(by="Conversion Rate", ascending=False).head(5)
    return render_template("dashboard.html", tables=[top_campaigns.to_html(classes='data')])

if __name__ == "__main__":
    app.run(debug=True)