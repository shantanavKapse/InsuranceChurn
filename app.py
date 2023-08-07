from flask import Flask, render_template, request, redirect
import pandas as pd
import plotly.express as px
import joblib

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

scaler = joblib.load('./minMaxScaler.pickle')
model = joblib.load('./rfClassifier.pickle')

@app.route('/', methods=['GET', 'POST'])
def home():
    data = None
    bar = None
    yes = False
    if request.method == "POST":
        file = request.files['csv_file']
        df = pd.read_csv(file)
        df.drop(columns=['Response', 'id'], inplace=True)
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)
        df['Predictions'] = model.predict(df)
        grp = df.groupby("Predictions")
        p, d, v = [], [], []
        for r in grp['Driving_License'].value_counts().index:
            p.append("Yes" if r[0] else "No")
            d.append("Yes" if r[1] else "No")
            v.append(grp['Driving_License'].value_counts()[r])
        dl_df = pd.DataFrame({"Driving License": d, "Prediction": p, "Churn": v})
        p, pi, v = [], [], []
        for r in grp['Previously_Insured'].value_counts().index:
            p.append("Yes" if r[0] else "No")
            pi.append("Yes" if r[1] else "No")
            v.append(grp['Previously_Insured'].value_counts()[r])
        pi_df = pd.DataFrame({"Previously Insured": pi, "Prediction": p, "Churn": v})
        p, g, v = [], [], []
        for r in grp['Gender'].value_counts().index:
            p.append("Yes" if r[0] else "No")
            g.append("Yes" if r[1] else "No")
            v.append(grp['Gender'].value_counts()[r])
        g_df = pd.DataFrame({"Gender": g, "Prediction": p, "Churn": v})
        bar = px.histogram(dl_df, x="Driving License", y="Churn", color="Prediction", barmode='group', template='plotly_dark').to_html()
        bar2 = px.histogram(pi_df, x="Previously Insured", y="Churn", color="Prediction", barmode='group', template='plotly_dark').to_html()
        bar3 = px.histogram(g_df, x="Gender", y="Churn", color="Prediction", barmode='group', template='plotly_dark').to_html()
        yes = True
        
    return render_template('index.html', data=df['Predictions'].replace({1.0: "Yes", 0.0: "No"}).value_counts(), bar_chart=bar, bar_chart2=bar2, bar_chart3=bar3, yes=yes)


if __name__ == '__main__':
    app.run(debug=True)