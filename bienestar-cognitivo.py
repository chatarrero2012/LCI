# 1. Configuración DEBE SER PRIMERO
import streamlit as st
st.set_page_config(
    page_title="Dashboard de Bienestar Cognitivo",
    page_icon="🧠",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.tuuniversidad.edu',
        'About': "Análisis predictivo del bienestar emocional"
    }
)

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
from groq import Groq
import edge_tts
import asyncio
import pygame
import re
import base64
def autoplay_audio(file_path):
     with open(file_path, "rb") as f:
        audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        st.audio(file_path, format="audio/mp3")
        st.warning("Presiona ▶️ en iPhone (restricciones de iOS)")
def generar_discurso_junta(analisis, destinatario="Padres de familia", tipo_comunicacion="Discurso"):
    """Versión corregida manteniendo tu estructura"""
    client = Groq(api_key="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")
    
    SYS_PROMPT = f"""
    Eres un coach educativo estilo TEDx. Genera un {tipo_comunicacion.lower()} que ELECTRIFIQUE a {destinatario.lower()} con:
1. **Apertura explosiva**: ¡Dato impactante + analogía memorable!  
   Ej: "Esto duele más que ver a un estudiante copiando en ChatGPT"  

2. **Hallazgos clave** (3 verdades incómodas con comparaciones crudas):  
   - "El 40% de nuestros estudiantes están más estresados que CEO en quiebra"  

3. **Recomendaciones de guerra** (2 acciones específicas):  
   - "No es otro taller... ¡es entrenamiento de supervivencia académica!"  

4. **Cierre épico**: Llamado a acción que erice la piel  

Reglas:  
- CERO lenguaje corporativo ("sinergias", "enfoque holístico")  
- 100% español neutro (nada de "móvil" → decir "celular")  
- Tono: Como si estuvieras salvando la educación a golpes
- No lo dividas por secciones como si todas las partes fueran un solo flujo de ideas bien compacto
    """
    
    USER_PROMPT = f"""
    Datos del análisis:
    - Hallazgos: {analisis['top_features']}
    - Riesgos: {analisis['riesgos']}
    - Correlación clave: {analisis['correlaciones']}
    """

    try:
        # Corrección clave: Asegurar que los parámetros se pasen por nombre
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Añadido explícitamente
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error en la API: {str(e)}")
        return None
# Integración en Streamlit
async def presentar_analisis_junta():
    st.title("🎙️ Discurso Automatizado para la Junta")
    
    # ==== NUEVOS SELECTORES (ÚNICO CAMBIO) ====
    col1, col2 = st.columns(2)
    with col1:
        destinatario = st.selectbox(
            "Dirigido a:",
            ["Padres de familia", "Junta directiva", "Estudiantes"],
            index=0
        )
    with col2:
        tipo_comunicacion = st.selectbox(
            "Tipo de comunicación:",
            ["Discurso", "Email", "Cartelera"],
            index=0
        )
    # ==========================================

    # ==== TODO EL RESTO DEL CÓDIGO PERMANECE EXACTAMENTE IGUAL ====
    df = load_data('Factor 1. Bienestar cognitivo emocional(1-54)-2.xlsx')
    
    with st.spinner("Analizando datos y preparando presentación..."):
        pivot_df = df.pivot_table(index='ID', columns='Pregunta', values='Puntaje')
        corr_matrix = pivot_df.corr()
        
        target_df = df.groupby('ID')['Puntaje'].mean().reset_index()
        features_df = pivot_df.fillna(0)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(features_df, target_df['Puntaje'])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)
        
        pca = PCA(n_components=3)
        components = pca.fit_transform(features_df)
        clusters = KMeans(n_clusters=3).fit_predict(components)
        
        analisis_completo = {
            "clusters": f"{len(np.unique(clusters))} grupos principales detectados",
            "top_features": features_df.columns[np.argsort(np.abs(shap_values).mean(0))[-3:]].tolist(),
            "correlaciones": f"Correlación más fuerte: {corr_matrix.stack().sort_values(ascending=False)[1:2].index[0]} (r={corr_matrix.stack().sort_values(ascending=False)[1:2].values[0]:.2f})",
            "riesgos": f"{np.mean(target_df['Puntaje'] < 2.5):.0%} estudiantes con puntaje bajo",
            "shap_values": f"{features_df.columns[np.argmax(np.abs(shap_values).mean(0))]} = Impacto más alto"
        }
        
        # Ajuste mínimo en la llamada para pasar los nuevos parámetros
        discurso = generar_discurso_junta(
            analisis_completo,
            destinatario=destinatario,  # Nuevo parámetro
            tipo_comunicacion=tipo_comunicacion  # Nuevo parámetro
        )
    
    # ==== EL RESTO DEL CÓDIGO SIGUE IGUAL ====
    if discurso:
        st.subheader("Borrador de Comunicación Generada:")
        with st.expander("Ver contenido completo", expanded=True):
            st.markdown(discurso)
            
        with st.expander("🔍 Ver datos de análisis utilizados"):
            st.write("**Clusters:**", analisis_completo['clusters'])
            st.write("**Variables más importantes:**", ", ".join(analisis_completo['top_features']))
            st.write("**Correlaciones clave:**", analisis_completo['correlaciones'])
            st.write("**Riesgos detectados:**", analisis_completo['riesgos'])
            st.write("**Impacto de variables (SHAP):**", analisis_completo['shap_values'])
            
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Descargar como PDF", discurso, file_name=f"comunicado_{destinatario[:3]}_{tipo_comunicacion[:3]}.pdf")
        with col2:
            #if st.button("🎧 Escuchar versión audio"):
                await generar_audio(discurso)
        
        st.write("---")
        st.subheader("✍️ Personalizar el discurso")
        nuevo_tono = st.selectbox("Ajustar tono:", ["Motivacional", "Urgente", "Empático"])
        if st.button("Generar nueva versión"):
            st.experimental_rerun()
            
    else:
        st.warning("No se pudo generar el contenido. Verifica los análisis previos.")
def limpiar_texto_para_voz(texto):
    # Elimina asteriscos, guiones raros, etc.
    texto_limpio = re.sub(r'[\*\#\_\-]', ' ', texto)  
    # Reduce múltiples espacios a uno solo
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
    return texto_limpio
async def generar_audio(texto):
    voice = "es-MX-DaliaNeural"  # Voz femenina expresiva
    clean_texto = limpiar_texto_para_voz(texto)
    communicate = edge_tts.Communicate(clean_texto, voice)
    await communicate.save("output.mp3")
    autoplay_audio("output.mp3")
    print("Audio guardado como 'output.mp3'")
# --------------------------------------------------
# 1. Carga y limpieza de datos (Formato profesional)
# --------------------------------------------------
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    
    # Extraer columnas de preguntas (asumiendo patrón: cada pregunta tiene 3 columnas)
    question_columns = [col for col in df.columns if 'Points - ' not in col and 'Feedback - ' not in col][5:]
    
    # Convertir a formato largo
    long_df = pd.melt(df, id_vars=['ID', 'Start time', 'Completion time'], 
                     value_vars=question_columns,
                     var_name='Pregunta', 
                     value_name='Respuesta')
    
    # Limpiar y convertir respuestas a numéricas
    long_df['Puntaje'] = long_df['Respuesta'].str.extract('Option (\d)').astype(float)
    
    return long_df.dropna(subset=['Puntaje'])

# --------------------------------------------
# 2. Análisis Exploratorio (Visualizaciones 3D)
# --------------------------------------------
def exploratory_analysis(df):
    # Heatmap interactivo de correlaciones
    pivot_df = df.pivot_table(index='ID', columns='Pregunta', values='Puntaje')
    fig = px.imshow(pivot_df.corr(), 
                   title='<b>Mapa de Calor: Correlación entre Variables</b>',
                   color_continuous_scale='RdBu',
                   width=1200, height=800)
    fig.update_layout(font=dict(size=12))
    fig.show()
    
    # Análisis de componentes principales 3D
    pca = PCA(n_components=3)
    components = pca.fit_transform(pivot_df.fillna(pivot_df.mean()))
    
    fig = px.scatter_3d(components, x=0, y=1, z=2,
                       color=KMeans(n_clusters=3).fit_predict(components),
                       title='<b>Segmentación de Respuestas en 3D (PCA)</b>',
                       labels={'0': 'PC1', '1': 'PC2', '2': 'PC3'},
                       width=1000, height=800)
    fig.update_traces(marker=dict(size=5))
    fig.show()

# ------------------------------------------
# 3. Modelo Predictivo con Interpretación AI (Versión Streamlit)
# ------------------------------------------
def build_ai_model(df):
    # Crear variable target (puntaje promedio por usuario)
    target_df = df.groupby('ID')['Puntaje'].mean().reset_index()
    features_df = df.pivot_table(index='ID', columns='Pregunta', values='Puntaje').fillna(0)
    
    # Entrenar modelo
    X_train, X_test, y_train, y_test = train_test_split(
        features_df,
        target_df['Puntaje'], 
        test_size=0.2,
        random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # SHAP para explicabilidad
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Configurar matplotlib para Streamlit
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')  # Necesario para entornos sin GUI
    
    # Visualización SHAP en Streamlit
    st.subheader('🕵️♂️ Interpretación AI: Factores Clave')
    with st.expander("Ver explicación del modelo", expanded=True):
        st.write("""
        **Cómo interpretar este gráfico:**
        - Variables ordenadas por impacto
        - Color: Valor de la característica
        - Posición X: Impacto en el puntaje
        """)
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(fig, clear_figure=True)
        
        st.markdown("""
        **Insights clave:**
        1. La variable más importante es **'{}'**
        2. **'{}'** muestra un efecto no lineal
        3. **'{}'** tiene impacto negativo sorprendente
        """.format(
            X_train.columns[np.argmax(np.abs(shap_values).mean(0))],
            X_train.columns[3],  # Ejemplo ajustable
            X_train.columns[1]   # Ejemplo ajustable
        ))

    # Métricas de rendimiento
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Entrenamiento", f"{model.score(X_train, y_train):.2%}")
    with col2:
        st.metric("R² Validación", f"{model.score(X_test, y_test):.2%}")
    with col3:
        st.metric("Error Promedio", f"{np.mean(np.abs(y_test - model.predict(X_test))):.2f} pts")

# --------------------------------------
# 4. Dashboard Interactivo con Streamlit (CORREGIDO)
# --------------------------------------
def create_dashboard(df):
    # Eliminar st.set_page_config() de aquí
    st.title('🔥 Análisis de Bienestar Cognitivo-Emocional - Dashboard Ejecutivo')
    
    # Filtros
    selected_questions = st.multiselect('Seleccionar Preguntas', df['Pregunta'].unique())
    
    # Gráficos dinámicos
    if selected_questions:
        filtered_df = df[df['Pregunta'].isin(selected_questions)]
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(filtered_df, x='Pregunta', y='Puntaje', 
                        color='Pregunta', title='Distribución de Puntajes')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(filtered_df, x='Puntaje', facet_col='Pregunta', 
                             nbins=4, title='Frecuencia de Respuestas')
            st.plotly_chart(fig, use_container_width=True)
    
    # Análisis temporal
    time_df = df.groupby([pd.Grouper(key='Completion time', freq='D'), 'Pregunta'])['Puntaje'].mean().reset_index()
    fig = px.line(time_df, x='Completion time', y='Puntaje', color='Pregunta',
                 title='Tendencia Temporal de Puntajes', markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# 5. Análisis de Clústeres
# ------------------------
def cluster_analysis(df):
    # 1. Preparar datos para clustering
    features_df = df.pivot_table(index='ID', columns='Pregunta', values='Puntaje', aggfunc='mean').fillna(0)
    
    # 2. Reducción de dimensionalidad optimizada
    pca = PCA(n_components=0.95)
    reduced_data = pca.fit_transform(features_df)
    n_components = pca.n_components_
    
    # 3. Crear DataFrame con nombres de columnas adecuados
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(reduced_data, columns=pca_columns)
    pca_df['ID'] = features_df.index
    pca_df['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(reduced_data)
    
    # 4. Visualización interactiva mejorada
    fig = px.scatter_matrix(
        pca_df,
        dimensions=pca_columns[:5],  # Solo primeras 5 componentes
        color='Cluster',
        title='<b>Análisis de Clústeres - Componentes Principales</b>',
        hover_data=['ID'],
        width=1200,
        height=800,
        opacity=0.7
    )
    
    # Personalización profesional
    fig.update_traces(
        marker=dict(
            size=5,
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        diagonal_visible=False
    )
    fig.update_layout(
        plot_bgcolor='rgba(240,240,240,0.9)',
        paper_bgcolor='white',
        font=dict(family="Arial", size=10)
    )
    
    fig.show()
    
    return df.merge(pca_df[['ID', 'Cluster']], on='ID', how='left')

# --------------------
# Ejecución Principal
# --------------------
if __name__ == "__main__":
    df = load_data('Factor 1. Bienestar cognitivo emocional(1-54)-2.xlsx')
    
    # Funciones síncronas
    exploratory_analysis(df)
    build_ai_model(df)
    clustered_df = cluster_analysis(df)
    create_dashboard(clustered_df)
    
    # Función asíncrona
    import asyncio
    asyncio.run(presentar_analisis_junta())
