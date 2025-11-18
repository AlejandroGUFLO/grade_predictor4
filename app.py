import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import plotly.graph_objects as go

# ------------------------------
# Load and prepare data
# ------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("proyectom.csv")
    
    # Target variables
    df["HighPerformance"] = (df["Calificaciones pasadas"] >= 9.2).astype(int)
    
    # Feature engineering
    df["eficiencia_estudio_pasado"] = df["Calificaciones pasadas"] / (df["Horas estudio pasadas "] + 1)
    df["carga_academica_pasada"] = df["Materias pasadas "] * df["Horas estudio pasadas "]
    df["carga_academica_actual"] = df["Materias nuevas"] * df["Horas de estudio actuales "]
    df["cambio_horas"] = df["Horas de estudio actuales "] - df["Horas estudio pasadas "]
    df["ratio_materias"] = df["Materias nuevas"] / (df["Materias pasadas "] + 1)
    
    return df

df = load_and_prepare_data()

# Features
feature_cols = [
    "Materias pasadas ",
    "Materias nuevas",
    "Calificaciones pasadas",
    "eficiencia_estudio_pasado",
    "carga_academica_actual",
    "ratio_materias"
]

X = df[feature_cols]

# Modelo de REGRESIÓN para predecir la calificación exacta
Y_grade = df["Calificaciones pasadas"]
scaler_reg = StandardScaler()
X_scaled_reg = scaler_reg.fit_transform(X)
model_regression = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
model_regression.fit(X_scaled_reg, Y_grade)

# Modelo de CLASIFICACIÓN para probabilidad de alto rendimiento
Y_class = df["HighPerformance"]
scaler_class = StandardScaler()
X_scaled_class = scaler_class.fit_transform(X)
model_classification = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model_classification.fit(X_scaled_class, Y_class)

# ------------------------------
# UI
# ------------------------------
st.title("🎓 Predictor de Calificaciones")
st.markdown("*Predice tu calificación esperada basada en tus hábitos de estudio*")

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
if st.button("🔮 Predecir Calificación", type="primary"):
    new_data = pd.DataFrame({
        "Materias pasadas ": [courses_past],
        "Materias nuevas": [courses_now],
        "Calificaciones pasadas": [grade_past],
        "eficiencia_estudio_pasado": [eficiencia],
        "carga_academica_actual": [carga_actual],
        "ratio_materias": [ratio_mat]
    })
    
    # Predicción de calificación
    new_data_scaled_reg = scaler_reg.transform(new_data)
    predicted_grade = model_regression.predict(new_data_scaled_reg)[0]
    
    # Predicción de probabilidad de alto rendimiento
    new_data_scaled_class = scaler_class.transform(new_data)
    probability = model_classification.predict_proba(new_data_scaled_class)[0][1]
    
    # Resultados
    st.markdown("---")
    st.subheader("📊 Resultado de la Predicción")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Determinar el color basado en la calificación
        grade_color = "🟢" if predicted_grade >= 9.2 else "🟡" if predicted_grade >= 8.5 else "🔴"
        st.metric(
            "Calificación Esperada", 
            f"{predicted_grade:.2f}",
            delta=f"{predicted_grade - grade_past:+.2f} vs semestre anterior"
        )
        st.markdown(f"### {grade_color}")
    
    with col2:
        st.metric(
            "Probabilidad Alto Rendimiento", 
            f"{probability*100:.1f}%",
            help="Probabilidad de obtener ≥9.2"
        )
    
    with col3:
        st.metric(
            "Eficiencia de Estudio",
            f"{eficiencia:.2f}",
            help="Calificación por hora de estudio"
        )
    
    # Gráfico tipo velocímetro para calificación
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = predicted_grade,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Calificación Esperada"},
        delta = {'reference': grade_past, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [6, 10], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [6, 7], 'color': "lightcoral"},
                {'range': [7, 8], 'color': "lightyellow"},
                {'range': [8, 9], 'color': "lightblue"},
                {'range': [9, 10], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9.2
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de cambio
    grade_change = predicted_grade - grade_past
    
    if grade_change > 0.3:
        st.success(f"📈 **¡Excelente!** Se espera una mejora de {grade_change:.2f} puntos")
    elif grade_change < -0.3:
        st.error(f"📉 **Atención:** Se espera una baja de {abs(grade_change):.2f} puntos")
    else:
        st.info(f"📊 **Estable:** Calificación similar al semestre anterior")
    
    # Recomendaciones basadas en la predicción
    st.subheader("💡 Recomendaciones")
    
    if predicted_grade < 9.0:
        st.warning("**Sugerencias para mejorar tu calificación:**")
        
        if eficiencia < 1.5:
            st.write("• 📚 **Eficiencia baja:** Tu aprovechamiento por hora es bajo. Prueba técnicas como:")
            st.write("  - Método Pomodoro (25 min estudio + 5 min descanso)")
            st.write("  - Estudio activo (resúmenes, mapas mentales)")
            st.write("  - Eliminar distracciones durante el estudio")
        
        if carga_actual > 80:
            st.write(f"• ⚠️ **Carga alta:** {courses_now} materias × {hours_now} horas = {carga_actual} (carga muy pesada)")
            st.write("  - Considera reducir una materia si es posible")
            st.write("  - Prioriza las materias más importantes")
        
        if hours_now < hours_past and grade_past >= 9.0:
            st.write(f"• ⏰ **Menos horas:** Pasaste de {hours_past}h a {hours_now}h semanales")
            st.write("  - Intenta mantener al menos las mismas horas de estudio")
        
        if grade_past < 8.5:
            st.write("• 🎯 **Historial bajo:** Considera buscar apoyo adicional:")
            st.write("  - Grupos de estudio")
            st.write("  - Tutorías o asesorías")
            st.write("  - Recursos en línea especializados")
    
    elif predicted_grade >= 9.2:
        st.success("**🌟 ¡Excelente proyección!**")
        st.write("• Mantén tus hábitos de estudio actuales")
        st.write("• Tu eficiencia de estudio es muy buena")
        st.write("• Considera ayudar a compañeros con dificultades")
    
    else:
        st.info("**✅ Buen camino**")
        st.write("• Estás cerca de alto rendimiento")
        st.write(f"• Solo necesitas {9.2 - predicted_grade:.2f} puntos más para llegar a 9.2")
        st.write("• Aumentar ligeramente tus horas de estudio podría ayudar")
    
    # Simulador: ¿Qué pasaría si cambio mis horas?
    st.subheader("🔄 Simulador: ¿Qué pasa si cambio mis horas de estudio?")
    
    hours_scenarios = []
    grades_scenarios = []
    
    for h in range(1, 21):
        sim_data = pd.DataFrame({
            "Materias pasadas ": [courses_past],
            "Materias nuevas": [courses_now],
            "Calificaciones pasadas": [grade_past],
            "eficiencia_estudio_pasado": [grade_past / (hours_past + 1)],
            "carga_academica_actual": [courses_now * h],
            "ratio_materias": [courses_now / (courses_past + 1)]
        })
        sim_scaled = scaler_reg.transform(sim_data)
        sim_grade = model_regression.predict(sim_scaled)[0]
        
        hours_scenarios.append(h)
        grades_scenarios.append(sim_grade)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hours_scenarios,
        y=grades_scenarios,
        mode='lines+markers',
        name='Calificación esperada',
        line=dict(color='steelblue', width=3),
        marker=dict(size=6)
    ))
    
    # Marcar el punto actual
    fig2.add_trace(go.Scatter(
        x=[hours_now],
        y=[predicted_grade],
        mode='markers',
        name='Tu situación actual',
        marker=dict(size=15, color='red', symbol='star')
    ))
    
    # Línea de referencia en 9.2
    fig2.add_hline(y=9.2, line_dash="dash", line_color="green", 
                   annotation_text="Alto rendimiento (9.2)")
    
    fig2.update_layout(
        title="Impacto de las horas de estudio en tu calificación",
        xaxis_title="Horas de estudio semanales",
        yaxis_title="Calificación esperada",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Importancia de variables
    st.subheader("📈 Factores más Importantes")
    
    feature_importance = pd.DataFrame({
        'Factor': ['Calificaciones pasadas', 'Eficiencia de estudio', 'Carga académica actual', 
                   'Materias anteriores', 'Materias actuales', 'Ratio de materias'],
        'Importancia': model_regression.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    fig3 = go.Figure(go.Bar(
        x=feature_importance['Importancia'],
        y=feature_importance['Factor'],
        orientation='h',
        marker=dict(color='steelblue')
    ))
    fig3.update_layout(
        title="¿Qué afecta más tu calificación?",
        xaxis_title="Importancia",
        height=300
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# Estadísticas del dataset
with st.expander("📊 Ver estadísticas del dataset"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estudiantes", len(df))
    with col2:
        st.metric("Calificación promedio", f"{df['Calificaciones pasadas'].mean():.2f}")
    with col3:
        st.metric("Alto rendimiento", f"{(Y_class.sum()/len(Y_class)*100):.1f}%")
    with col4:
        st.metric("Horas promedio", f"{df['Horas de estudio actuales '].mean():.1f}")
