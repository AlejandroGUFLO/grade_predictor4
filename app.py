
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# ------------------------------
# Load and prepare data
# ------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("proyectom.csv")
    
    # Target variable
    df["HighPerformance"] = (df["Calificaciones pasadas"] >= 9.2).astype(int)
    
    # Feature engineering - crear variables más significativas
    df["eficiencia_estudio_pasado"] = df["Calificaciones pasadas"] / (df["Horas estudio pasadas "] + 1)
    df["carga_academica_pasada"] = df["Materias pasadas "] * df["Horas estudio pasadas "]
    df["carga_academica_actual"] = df["Materias nuevas"] * df["Horas de estudio actuales "]
    df["cambio_horas"] = df["Horas de estudio actuales "] - df["Horas estudio pasadas "]
    df["ratio_materias"] = df["Materias nuevas"] / (df["Materias pasadas "] + 1)
    
    return df

df = load_and_prepare_data()

# Selección de features mejoradas
feature_cols = [
    "Materias pasadas ",
    "Materias nuevas",
    "Calificaciones pasadas",
    "eficiencia_estudio_pasado",
    "carga_academica_actual",
    "ratio_materias"
]

X = df[feature_cols]
Y = df["HighPerformance"]

# Normalización de features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Usar Random Forest en lugar de Regresión Logística
# Es más robusto y captura relaciones no lineales
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_scaled, Y)

# ------------------------------
# UI
# ------------------------------
st.title("🎓 Predictor de Rendimiento Académico")
st.markdown("*Predicción mejorada con análisis de eficiencia de estudio*")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Semestre Anterior")
    courses_past = st.number_input("Materias cursadas", min_value=1, max_value=15, value=7, key="cp")
    hours_past = st.number_input("Horas de estudio semanales", min_value=1, max_value=30, value=5, key="hp")
    grade_past = st.number_input("Calificación final", min_value=6.0, max_value=10.0, value=9.0, step=0.1, key="gp")

with col2:
    st.subheader("📖 Semestre Actual")
    courses_now = st.number_input("Materias cursando", min_value=1, max_value=15, value=8, key="cn")
    hours_now = st.number_input("Horas de estudio semanales", min_value=1, max_value=30, value=5, key="hn")

# ------------------------------
# Cálculo de features derivadas
# ------------------------------
eficiencia = grade_past / (hours_past + 1)
carga_actual = courses_now * hours_now
ratio_mat = courses_now / (courses_past + 1)

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔮 Predecir Rendimiento", type="primary"):
    new_data = pd.DataFrame({
        "Materias pasadas ": [courses_past],
        "Materias nuevas": [courses_now],
        "Calificaciones pasadas": [grade_past],
        "eficiencia_estudio_pasado": [eficiencia],
        "carga_academica_actual": [carga_actual],
        "ratio_materias": [ratio_mat]
    })
    
    new_data_scaled = scaler.transform(new_data)
    
    prediction = model.predict(new_data_scaled)[0]
    probability = model.predict_proba(new_data_scaled)[0][1]
    
    # Resultados
    st.markdown("---")
    st.subheader("📊 Resultado de la Predicción")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicción", 
            "Alto ≥9.2" if prediction == 1 else "Bajo <9.2",
            delta="Buen rendimiento" if prediction == 1 else "Mejorar hábitos"
        )
    
    with col2:
        st.metric(
            "Probabilidad", 
            f"{probability*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Eficiencia de Estudio",
            f"{eficiencia:.2f}",
            help="Calificación por hora de estudio"
        )
    
    # Gráfico de probabilidad
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad de Alto Rendimiento"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "lightyellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recomendaciones
    st.subheader("💡 Recomendaciones")
    
    if prediction == 0:
        st.warning("**Sugerencias para mejorar:**")
        if eficiencia < 1.5:
            st.write("• Tu eficiencia de estudio es baja. Prueba técnicas como Pomodoro o estudio activo")
        if carga_actual > 80:
            st.write("• Tu carga académica es muy alta. Considera optimizar tu tiempo")
        if hours_now < hours_past and grade_past >= 9.0:
            st.write("• Estás estudiando menos que antes. Mantén al menos las mismas horas")
    else:
        st.success("**¡Vas por buen camino!**")
        st.write("• Mantén tus hábitos de estudio actuales")
        st.write("• Tu eficiencia de estudio es buena")
    
    # Importancia de variables
    st.subheader("📈 Factores más Importantes")
    
    feature_importance = pd.DataFrame({
        'Factor': ['Calificaciones pasadas', 'Eficiencia de estudio', 'Carga académica actual', 
                   'Materias anteriores', 'Materias actuales', 'Ratio de materias'],
        'Importancia': model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    fig2 = go.Figure(go.Bar(
        x=feature_importance['Importancia'],
        y=feature_importance['Factor'],
        orientation='h',
        marker=dict(color='steelblue')
    ))
    fig2.update_layout(
        title="¿Qué afecta más tu rendimiento?",
        xaxis_title="Importancia",
        height=300
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# Estadísticas del dataset
with st.expander("📊 Ver estadísticas del dataset"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estudiantes analizados", len(df))
    with col2:
        st.metric("Alto rendimiento (≥9.2)", f"{(Y.sum()/len(Y)*100):.1f}%")
    with col3:
        st.metric("Calificación promedio", f"{df['Calificaciones pasadas'].mean():.2f}")
