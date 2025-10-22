"""
Sistema de Asistencia con Reconocimiento Facial
Version Simplificada - Todo en un archivo
Usando: Streamlit + DeepFace + OpenCV + JSON
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime, date
import pandas as pd
import json
import os
from PIL import Image
import base64
from deepface import DeepFace
import shutil

# ============================================
# CONFIGURACIÓN
# ============================================
ESTUDIANTES_FILE = "estudiantes.json"
ASISTENCIAS_FILE = "asistencias.json"
FOTOS_DIR = "fotos_estudiantes"
THRESHOLD = 0.10  # Umbral de similitud (menor = más estricto)

# Crear carpetas si no existen
if not os.path.exists(FOTOS_DIR):
    os.makedirs(FOTOS_DIR)

# ============================================
# FUNCIONES DE DATOS (JSON)
# ============================================

def cargar_estudiantes():
    """Cargar estudiantes desde JSON"""
    if os.path.exists(ESTUDIANTES_FILE):
        with open(ESTUDIANTES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def guardar_estudiantes(estudiantes):
    """Guardar estudiantes en JSON"""
    with open(ESTUDIANTES_FILE, 'w', encoding='utf-8') as f:
        json.dump(estudiantes, f, indent=4, ensure_ascii=False)

def cargar_asistencias():
    """Cargar asistencias desde JSON"""
    if os.path.exists(ASISTENCIAS_FILE):
        with open(ASISTENCIAS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def guardar_asistencias(asistencias):
    """Guardar asistencias en JSON"""
    with open(ASISTENCIAS_FILE, 'w', encoding='utf-8') as f:
        json.dump(asistencias, f, indent=4, ensure_ascii=False)

def agregar_estudiante(codigo, nombre, apellido, email, foto_path):
    """Agregar nuevo estudiante"""
    estudiantes = cargar_estudiantes()
    
    # Verificar si ya existe
    for est in estudiantes:
        if est['codigo'] == codigo:
            return False, "El código ya existe"
    
    nuevo_estudiante = {
        'codigo': codigo,
        'nombre': nombre,
        'apellido': apellido,
        'email': email,
        'foto_path': foto_path,
        'fecha_registro': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    estudiantes.append(nuevo_estudiante)
    guardar_estudiantes(estudiantes)
    return True, "Estudiante registrado"

def registrar_asistencia(codigo, nombre, apellido, similitud):
    """Registrar asistencia de un estudiante"""
    asistencias = cargar_asistencias()
    fecha_hoy = date.today().strftime('%Y-%m-%d')
    hora_actual = datetime.now().strftime('%H:%M:%S')
    
    # Verificar si ya registró hoy
    for asist in asistencias:
        if asist['codigo'] == codigo:
            return False, "Ya registró asistencia"
    
    nueva_asistencia = {
        'codigo': codigo,
        'nombre': nombre,
        'apellido': apellido,
        'fecha': fecha_hoy,
        'hora': hora_actual,
        'similitud': f"{(1-similitud)*100:.2f}%"
    }
    
    asistencias.append(nueva_asistencia)
    guardar_asistencias(asistencias)
    return True, "Asistencia registrada"

def obtener_asistencias_hoy():
    """Obtener asistencias del día actual"""
    asistencias = cargar_asistencias()
    fecha_hoy = date.today().strftime('%Y-%m-%d')
    return [a for a in asistencias if a['fecha'] == fecha_hoy]

# ============================================
# FUNCIONES DE RECONOCIMIENTO FACIAL
# ============================================

def detectar_rostro(imagen):
    """Detectar rostros en una imagen usando OpenCV"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gris, 1.3, 5)
    return len(rostros) > 0, len(rostros), rostros

def reconocer_rostro(imagen_test):
    """Reconocer rostro comparando con todos los estudiantes registrados"""
    estudiantes = cargar_estudiantes()
    
    if len(estudiantes) == 0:
        return False, None, None, "No hay estudiantes registrados"
    
    # Guardar imagen temporal
    temp_path = "temp_test.jpg"
    cv2.imwrite(temp_path, imagen_test)
    
    mejor_match = None
    mejor_distancia = float('inf')
    
    try:
        for estudiante in estudiantes:
            try:
                # Comparar rostros usando DeepFace
                resultado = DeepFace.verify(
                    img1_path=temp_path,
                    img2_path=estudiante['foto_path'],
                    model_name='VGG-Face',
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                distancia = resultado['distance']
                
                # Si la distancia es menor que el threshold y es la mejor hasta ahora
                if distancia < THRESHOLD and distancia < mejor_distancia:
                    mejor_distancia = distancia
                    mejor_match = estudiante
                    
            except Exception as e:
                print(f"Error comparando con {estudiante['codigo']}: {e}")
                continue
        
        # Eliminar archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if mejor_match:
            return True, mejor_match, mejor_distancia, "Match encontrado"
        else:
            return False, None, None, "No se encontró coincidencia"
            
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, None, None, f"Error: {str(e)}"

def capturar_desde_camara():
    """Capturar foto desde la cámara"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return False, "No se puede acceder a la cámara"
    
    # Dar tiempo a la cámara para ajustarse
    for _ in range(10):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, "Error al capturar imagen"
    
    return True, frame

# ============================================
# INTERFAZ DE STREAMLIT
# ============================================

st.set_page_config(
    page_title="Sistema de Asistencia Facial",
    page_icon="📸",
    layout="wide"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📸 Sistema de Asistencia Facial</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Menú")
opcion = st.sidebar.radio(
    "Selecciona:",
    ["🏠 Inicio", "➕ Registrar", "✅ Asistencia", "📊 Reportes"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"📅 {datetime.now().strftime('%d/%m/%Y')}\n\n⏰ {datetime.now().strftime('%H:%M:%S')}")

# ============================================
# PÁGINA INICIO
# ============================================
if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema")
    
    estudiantes = cargar_estudiantes()
    asistencias_hoy = obtener_asistencias_hoy()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 👥 Estudiantes")
        st.metric("Total", len(estudiantes))
    
    with col2:
        st.success("### ✅ Asistencias Hoy")
        st.metric("Total", len(asistencias_hoy))
    
    with col3:
        st.warning("### 📊 Porcentaje")
        if len(estudiantes) > 0:
            porcentaje = (len(asistencias_hoy) / len(estudiantes)) * 100
            st.metric("Asistencia", f"{porcentaje:.1f}%")
        else:
            st.metric("Asistencia", "0%")
    
    st.markdown("---")
    
    st.subheader("📋 Instrucciones")
    
    st.info("""
    **➕ Registrar Estudiante:**
    1. Ve al menú "Registrar"
    2. Completa los datos del estudiante
    3. Captura una foto clara del rostro
    4. Guarda el registro
    
    **✅ Tomar Asistencia:**
    1. Ve al menú "Asistencia"
    2. Captura foto del estudiante
    3. El sistema reconocerá automáticamente
    4. Se registrará la asistencia
    
    **📊 Ver Reportes:**
    - Consulta asistencias del día
    - Exporta datos a CSV
    """)

# ============================================
# PÁGINA REGISTRAR
# ============================================
elif opcion == "➕ Registrar":
    st.header("Registrar Nuevo Estudiante")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Datos del Estudiante")
        
        codigo = st.text_input("Código*", placeholder="Ej: EST001")
        nombre = st.text_input("Nombre*", placeholder="Ej: Juan")
        apellido = st.text_input("Apellido*", placeholder="Ej: Pérez")
        #email = st.text_input("Email", placeholder="juan@email.com")
    
    with col2:
        st.subheader("📸 Foto del Estudiante")
        
        metodo = st.radio("Método:", ["📷 Capturar Cámara", "📁 Subir Imagen"])
        
        if metodo == "📷 Capturar Cámara":
            if st.button("📷 Capturar", width='content'):
                with st.spinner("Capturando..."):
                    exito, resultado = capturar_desde_camara()
                    
                    if exito:
                        st.session_state['foto_registro'] = resultado
                        st.success("✅ Foto capturada")
                    else:
                        st.error(f"❌ {resultado}")
        else:
            archivo = st.file_uploader("Selecciona imagen", type=['jpg', 'jpeg', 'png'])
            if archivo:
                imagen = Image.open(archivo)
                imagen_np = np.array(imagen)
                # Convertir RGB a BGR para OpenCV
                if len(imagen_np.shape) == 3:
                    imagen_np = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)
                st.session_state['foto_registro'] = imagen_np
        
        # Mostrar foto si existe
        if 'foto_registro' in st.session_state:
            foto = st.session_state['foto_registro']
            
            # Detectar rostro
            tiene_rostro, cantidad, rostros = detectar_rostro(foto)
            
            # Dibujar rectángulos
            foto_mostrar = foto.copy()
            for (x, y, w, h) in rostros:
                cv2.rectangle(foto_mostrar, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Convertir BGR a RGB para mostrar
            foto_rgb = cv2.cvtColor(foto_mostrar, cv2.COLOR_BGR2RGB)
            st.image(foto_rgb, caption=f"Rostros: {cantidad}", width='content')
            
            if cantidad == 1:
                st.success("✅ Perfecto!")
            elif cantidad == 0:
                st.error("❌ No se detectó rostro")
            else:
                st.warning(f"⚠️ Se detectaron {cantidad} rostros. Debe haber solo 1")
    
    st.markdown("---")
    
    # Botón de registro
    if st.button("✅ REGISTRAR ESTUDIANTE", type="primary", width='content'):
        if not codigo or not nombre or not apellido:
            st.error("❌ Completa todos los campos obligatorios")
        elif 'foto_registro' not in st.session_state:
            st.error("❌ Captura una foto del estudiante")
        else:
            # Verificar que tenga un solo rostro
            tiene_rostro, cantidad, _ = detectar_rostro(st.session_state['foto_registro'])
            
            if not tiene_rostro:
                st.error("❌ No se detectó ningún rostro en la imagen")
            elif cantidad > 1:
                st.error(f"❌ Se detectaron {cantidad} rostros. Debe haber solo uno")
            else:
                # Guardar foto
                foto_filename = f"{codigo}.jpg"
                foto_path = os.path.join(FOTOS_DIR, foto_filename)
                cv2.imwrite(foto_path, st.session_state['foto_registro'])
                
                # Registrar estudiante
                exito, mensaje = agregar_estudiante(codigo, nombre, apellido, "user@ciaf.edu.co", foto_path)
                
                if exito:
                    st.success(f"✅ {mensaje}")
                    st.balloons()
                    del st.session_state['foto_registro']
                    st.rerun()
                else:
                    st.error(f"❌ {mensaje}")

# ============================================
# PÁGINA ASISTENCIA
# ============================================
elif opcion == "✅ Asistencia":
    st.header("Tomar Asistencia")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Capturar Rostro")
        
        metodo = st.radio("Método:", ["📷 Cámara", "📁 Imagen"], key="asist_metodo")
        
        if metodo == "📷 Cámara":
            if st.button("📷 Capturar Foto", width='content'):
                with st.spinner("Capturando..."):
                    exito, resultado = capturar_desde_camara()
                    
                    if exito:
                        st.session_state['foto_asistencia'] = resultado
                        st.success("✅ Capturada")
                    else:
                        st.error(f"❌ {resultado}")
        else:
            archivo = st.file_uploader("Imagen", type=['jpg', 'jpeg', 'png'], key="asist_upload")
            if archivo:
                imagen = Image.open(archivo)
                imagen_np = np.array(imagen)
                if len(imagen_np.shape) == 3:
                    imagen_np = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)
                st.session_state['foto_asistencia'] = imagen_np
        
        # Mostrar foto
        if 'foto_asistencia' in st.session_state:
            foto = st.session_state['foto_asistencia']
            tiene_rostro, cantidad, rostros = detectar_rostro(foto)
            
            foto_mostrar = foto.copy()
            for (x, y, w, h) in rostros:
                cv2.rectangle(foto_mostrar, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            foto_rgb = cv2.cvtColor(foto_mostrar, cv2.COLOR_BGR2RGB)
            st.image(foto_rgb, caption=f"Rostros: {cantidad}", width='content')

            # Botón para limpiar foto
            if st.button("🗑️ Limpiar Foto", use_container_width=True, key="limpiar_asist"):
                del st.session_state['foto_asistencia']
                st.rerun()
    
    with col2:
        st.subheader("🔍 Resultado")
        
        if st.button("✅ RECONOCER Y REGISTRAR", type="primary", width='content'):
            if 'foto_asistencia' not in st.session_state:
                st.error("❌ Captura una foto primero")
            else:
                with st.spinner("Reconociendo rostro..."):
                    # Reconocer
                    encontrado, estudiante, distancia, mensaje = reconocer_rostro(
                        st.session_state['foto_asistencia']
                    )
                    
                    if encontrado:
                        # Registrar asistencia
                        exito, msg = registrar_asistencia(
                            estudiante['codigo'],
                            estudiante['nombre'],
                            estudiante['apellido'],
                            distancia
                        )
                        
                        if exito:
                            similitud = (1 - distancia) * 100
                            st.success(f"""
                            ### ✅ ASISTENCIA REGISTRADA
                            
                            **Estudiante:** {estudiante['nombre']} {estudiante['apellido']}  
                            **Código:** {estudiante['codigo']}  
                            **Similitud:** {similitud:.2f}%  
                            **Hora:** {datetime.now().strftime('%H:%M:%S')}
                            """)
                            st.balloons()
                            del st.session_state['foto_asistencia']
                        else:
                            st.warning(f"⚠️ {msg}")
                    else:
                        st.error(f"❌ {mensaje}")
        
        st.markdown("---")
        st.subheader("📋 Asistencias Hoy")
        
        asistencias = obtener_asistencias_hoy()
        
        if len(asistencias) > 0:
            df = pd.DataFrame(asistencias)
            st.dataframe(df, width='content', hide_index=True)
        else:
            st.info("ℹ️ Sin asistencias registradas hoy")

# ============================================
# PÁGINA REPORTES
# ============================================
elif opcion == "📊 Reportes":
    st.header("Reportes y Consultas")
    
    tab1, tab2 = st.tabs(["📅 Hoy", "👥 Estudiantes"])
    
    with tab1:
        st.subheader("Asistencias de Hoy")
        
        asistencias = obtener_asistencias_hoy()
        
        if len(asistencias) > 0:
            df = pd.DataFrame(asistencias)
            st.dataframe(df, width='content', hide_index=True)
            
            # Exportar
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar CSV",
                csv,
                f"asistencia_{date.today().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
            
            # Stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", len(asistencias))
            with col2:
                total_est = len(cargar_estudiantes())
                if total_est > 0:
                    pct = (len(asistencias) / total_est) * 100
                    st.metric("% Asistencia", f"{pct:.1f}%")
        else:
            st.info("ℹ️ No hay asistencias hoy")
    
    with tab2:
        st.subheader("Lista de Estudiantes")
        
        estudiantes = cargar_estudiantes()
        
        if len(estudiantes) > 0:
            df = pd.DataFrame(estudiantes)
            df = df[['codigo', 'nombre', 'apellido', 'email', 'fecha_registro']]
            st.dataframe(df, width='content', hide_index=True)
            
            st.metric("Total Estudiantes", len(estudiantes))
            
            # Exportar
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar Lista",
                csv,
                "estudiantes.csv",
                "text/csv"
            )
        else:
            st.info("ℹ️ No hay estudiantes registrados")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Sistema de Asistencia v2.0 - DeepFace + OpenCV + JSON</p>
</div>
""", unsafe_allow_html=True)