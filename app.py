import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Smart City Traffic Predictor", layout="wide")

# 1. CHARGEMENT DES FICHIERS
@st.cache_resource
def load_assets():
    model = joblib.load('traffic_model.pkl')
    features = joblib.load('features_list.pkl')
    return model, features

try:
    model, features_list = load_assets()
except:
    st.error("Erreur : Les fichiers .pkl sont introuvables. Exécutez d'abord le Notebook.")
    st.stop()

# 2. INTERFACE LATÉRALE (SIDEBAR) - Entrées utilisateur
st.sidebar.header("📊 Paramètres de Simulation")
st.sidebar.write("Modifiez les paramètres pour prédire le trafic.")

def user_input_features():
    temp = st.sidebar.slider("Température (K)", 240, 315, 280)
    hour = st.sidebar.slider("Heure de la journée", 0, 23, 12)
    day = st.sidebar.selectbox("Jour de la semaine", 
                               options=[0, 1, 2, 3, 4, 5, 6], 
                               format_func=lambda x: ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"][x])
    month = st.sidebar.slider("Mois", 1, 12, 6)
    
    st.sidebar.subheader("Historique (Lags)")
    lag_1h = st.sidebar.number_input("Trafic il y a 1h", value=3000)
    lag_24h = st.sidebar.number_input("Trafic hier même heure", value=3200)
    lag_168h = st.sidebar.number_input("Trafic semaine dernière même heure", value=3100)
    
    # Autres valeurs simplifiées pour la démo
    data = {
        'temp': temp,
        'hour': hour,
        'day_of_week': day,
        'month': month,
        'lag_1h': lag_1h,
        'lag_2h': lag_1h * 0.9, # Simplification
        'lag_24h': lag_24h,
        'lag_168h': lag_168h,
        'rolling_mean_24h': (lag_1h + lag_24h) / 2
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# 3. PAGE PRINCIPALE
st.title("🚦 Prédiction du Trafic Urbain - Smart City")
st.markdown("""
Cette application utilise un modèle **XGBoost** pour prédire le volume de trafic sur une autoroute urbaine 
en fonction de la météo, de l'heure et de l'historique récent.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔮 Résultat de la Prédiction")
    prediction = model.predict(input_df[features_list])[0]
    
    # Affichage du score principal
    st.metric(label="Volume de trafic estimé", value=f"{int(prediction)} véhicules / heure")
    
    # Jauge visuelle
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Intensité du trafic"},
        gauge = {
            'axis': {'range': [0, 8000]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 2000], 'color': "green"},
                {'range': [2000, 5000], 'color': "orange"},
                {'range': [5000, 8000], 'color': "red"}],
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.subheader("💡 Explication des Variables")
    # Importance des variables
    importance = pd.DataFrame({
        'Feature': features_list,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h',
                     title="Quels facteurs influencent le plus le modèle ?",
                     color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig_imp, use_container_width=True)

# 4. GRAPHIQUE DE TENDANCE (SIMULATION)
st.divider()
st.subheader("📈 Simulation de la tendance sur 24h")

# Création d'un mini dataset pour 24h basé sur les réglages
hours = list(range(24))
sim_data = []
for h in hours:
    temp_row = input_df.copy()
    temp_row['hour'] = h
    # On ajuste un peu le trafic pour faire une courbe réaliste
    if 7 <= h <= 9 or 16 <= h <= 19:
        temp_row['lag_1h'] += 1000 # Simulation d'heure de pointe
    sim_data.append(model.predict(temp_row[features_list])[0])

fig_trend = px.line(x=hours, y=sim_data, labels={'x': 'Heure', 'y': 'Trafic prédit'},
                    title="Évolution prévue du trafic sur la journée",
                    markers=True)
fig_trend.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.1, annotation_text="Pointe Matin")
fig_trend.add_vrect(x0=17, x1=19, fillcolor="red", opacity=0.1, annotation_text="Pointe Soir")
st.plotly_chart(fig_trend, use_container_width=True)

st.info("ℹ️ Le modèle est particulièrement sensible aux 'Lags' (trafic passé) et à l'heure, car le trafic urbain est très cyclique.")


### to lunch  streamlit run app.py