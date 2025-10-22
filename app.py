"""
Sistema de Asistencia con Reconocimiento Facial EN TIEMPO REAL
Optimizado para fila de estudiantes - Solo poner la cara y pasar
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime, date
import pandas as pd
import json
import os
from PIL import Image
from deepface import DeepFace
import time
import threading

# ============================================
# CONFIGURACIÓN
# ============================================
ESTUDIANTES_FILE = "estudiantes.json"
ASISTENCIAS_FILE = "asistencias.json"
FOTOS_DIR = "fotos_estudiantes"
THRESHOLD = 0.10  # 90% de similitud
RECONOCIMIENTO_COOLDOWN = 3  # Segundos entre reconocimientos del mismo estudiante

# Crear carpetas si no existen
if not os.path.exists(FOTOS_DIR):
    os.makedirs(FOTOS_DIR)

# Variables globales para la cámara en tiempo real
camera_active = False
last_recognition_time = {}

# ============================================
# FUNCIONES DE DATOS (JSON)
# ============================================

def cargar_estudiantes():
    if os.path.exists(ESTUDIANTES_FILE):
        with open(ESTUDIANTES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def guardar_estudiantes(estudiantes):
    with open(ESTUDIANTES_FILE, 'w', encoding='utf-8') as f:
        json.dump(estudiantes, f, indent=4, ensure_ascii=False)

def cargar_asistencias():
    if os.path.exists(ASISTENCIAS_FILE):
        with open(ASISTENCIAS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def guardar_asistencias(asistencias):
    with open(ASISTENCIAS_FILE, 'w', encoding='utf-8') as f:
        json.dump(asistencias, f, indent=4, ensure_ascii=False)

def agregar_estudiante(codigo, nombre, apellido, email, foto_path):
    estudiantes = cargar_estudiantes()
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
    asistencias = cargar_asistencias()
    fecha_hoy = date.today().strftime('%Y-%m-%d')
    hora_actual = datetime.now().strftime('%H:%M:%S')
    
    # Verificar si ya registró hoy
    for asist in asistencias:
        if asist['codigo'] == codigo and asist['fecha'] == fecha_hoy:
            return False, "Ya registró hoy"
    
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
    asistencias = cargar_asistencias()
    fecha_hoy = date.today().strftime('%Y-%m-%d')
    return [a for a in asistencias if a['fecha'] == fecha_hoy]

def puede_reconocer(codigo):
    """Verifica si pasó suficiente tiempo desde el último reconocimiento"""
    global last_recognition_time
    ahora = time.time()
    
    if codigo not in last_recognition_time:
        last_recognition_time[codigo] = ahora
        return True
    
    tiempo_transcurrido = ahora - last_recognition_time[codigo]
    if tiempo_transcurrido >= RECONOCIMIENTO_COOLDOWN:
        last_recognition_time[codigo] = ahora
        return True
    
    return False

# ============================================
# FUNCIONES DE RECONOCIMIENTO FACIAL
# ============================================

def detectar_rostro(imagen):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gris, 1.3, 5)
    return len(rostros) > 0, len(rostros), rostros

def reconocer_rostro_rapido(imagen_test):
    """Versión optimizada para reconocimiento en tiempo real"""
    estudiantes = cargar_estudiantes()
    
    if len(estudiantes) == 0:
        return False, None, None, "No hay estudiantes"
    
    temp_path = "temp_realtime.jpg"
    cv2.imwrite(temp_path, imagen_test)
    
    mejor_match = None
    mejor_distancia = float('inf')
    
    try:
        for estudiante in estudiantes:
            try:
                resultado = DeepFace.verify(
                    img1_path=temp_path,
                    img2_path=estudiante['foto_path'],
                    model_name='VGG-Face',
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                distancia = resultado['distance']
                
                if distancia < THRESHOLD and distancia < mejor_distancia:
                    mejor_distancia = distancia
                    mejor_match = estudiante
                    
            except Exception as e:
                continue
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if mejor_match:
            return True, mejor_match, mejor_distancia, "Match"
        else:
            return False, None, None, "No match"
            
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, None, None, f"Error: {str(e)}"

def capturar_desde_camara():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "No se puede acceder a la cámara"
    
    for _ in range(10):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, "Error al capturar"
    
    return True, frame

# ============================================
# INTERFAZ DE STREAMLIT
# ============================================

st.set_page_config(
    page_title="Asistencia Facial - Tiempo Real",
    page_icon="📸",
    layout="wide"
)

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
    .big-button {
        font-size: 1.5rem !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
    .success-banner {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
        animation: slideIn 0.5s;
    }
    @keyframes slideIn {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📸 Sistema de Asistencia - Tiempo Real</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Menú")
opcion = st.sidebar.radio(
    "Selecciona:",
    ["🏠 Inicio", "➕ Registrar", "⚡ Asistencia Tiempo Real", "📊 Reportes"]
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
    
    st.subheader("🚀 Modo Tiempo Real")
    
    st.success("""
    ### ⚡ Sistema Optimizado para Filas
    
    **Cómo funciona:**
    1. Ve a "Asistencia Tiempo Real"
    2. Activa la cámara
    3. Los estudiantes solo ponen su cara frente a la cámara
    4. El sistema reconoce automáticamente y registra
    5. Siguiente estudiante → Repetir paso 3
    
    **Velocidad:** ~2-5 segundos por estudiante
    **Cooldown:** 3 segundos entre reconocimientos
    """)

# ============================================
# PÁGINA REGISTRAR
# ============================================
elif opcion == "➕ Registrar":
    st.header("Registrar Nuevo Estudiante")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Datos del Estudiante")
        
        codigo = st.text_input("Código*", placeholder="EST001")
        nombre = st.text_input("Nombre*", placeholder="Juan")
        apellido = st.text_input("Apellido*", placeholder="Pérez")
        email = st.text_input("Email", placeholder="juan@email.com")
    
    with col2:
        st.subheader("📸 Foto del Estudiante")
        
        metodo = st.radio("Método:", ["📷 Cámara", "📁 Imagen"])
        
        if metodo == "📷 Cámara":
            if st.button("📷 Capturar", use_container_width=True):
                with st.spinner("Capturando..."):
                    exito, resultado = capturar_desde_camara()
                    if exito:
                        st.session_state['foto_registro'] = resultado
                        st.success("✅ Capturada")
                    else:
                        st.error(f"❌ {resultado}")
        else:
            archivo = st.file_uploader("Imagen", type=['jpg', 'jpeg', 'png'])
            if archivo:
                imagen = Image.open(archivo)
                imagen_np = np.array(imagen)
                if len(imagen_np.shape) == 3:
                    imagen_np = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)
                st.session_state['foto_registro'] = imagen_np
        
        if 'foto_registro' in st.session_state:
            foto = st.session_state['foto_registro']
            tiene_rostro, cantidad, rostros = detectar_rostro(foto)
            
            foto_mostrar = foto.copy()
            for (x, y, w, h) in rostros:
                cv2.rectangle(foto_mostrar, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            foto_rgb = cv2.cvtColor(foto_mostrar, cv2.COLOR_BGR2RGB)
            st.image(foto_rgb, caption=f"Rostros: {cantidad}", width=400)
            
            if cantidad == 1:
                st.success("✅ Perfecto!")
            elif cantidad == 0:
                st.error("❌ No se detectó rostro")
            else:
                st.warning(f"⚠️ {cantidad} rostros. Debe haber 1")
            
            if st.button("🗑️ Limpiar", use_container_width=True):
                del st.session_state['foto_registro']
                st.rerun()
    
    st.markdown("---")
    
    if st.button("✅ REGISTRAR", type="primary", use_container_width=True):
        if not codigo or not nombre or not apellido:
            st.error("❌ Completa campos obligatorios")
        elif 'foto_registro' not in st.session_state:
            st.error("❌ Captura una foto")
        else:
            tiene_rostro, cantidad, _ = detectar_rostro(st.session_state['foto_registro'])
            
            if not tiene_rostro:
                st.error("❌ No se detectó rostro")
            elif cantidad > 1:
                st.error(f"❌ {cantidad} rostros. Debe haber 1")
            else:
                foto_filename = f"{codigo}.jpg"
                foto_path = os.path.join(FOTOS_DIR, foto_filename)
                cv2.imwrite(foto_path, st.session_state['foto_registro'])
                
                exito, mensaje = agregar_estudiante(codigo, nombre, apellido, email, foto_path)
                
                if exito:
                    st.success(f"✅ {mensaje}")
                    st.balloons()
                    del st.session_state['foto_registro']
                    st.rerun()
                else:
                    st.error(f"❌ {mensaje}")

# ============================================
# PÁGINA ASISTENCIA TIEMPO REAL
# ============================================
elif opcion == "⚡ Asistencia Tiempo Real":
    st.header("🎥 Reconocimiento en Tiempo Real")
    
    # Verificar que haya estudiantes registrados
    estudiantes = cargar_estudiantes()
    if len(estudiantes) == 0:
        st.warning("⚠️ No hay estudiantes registrados. Ve a 'Registrar' primero.")
        st.stop()
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📹 Cámara en Vivo")
        
        # Placeholder para la cámara
        FRAME_WINDOW = st.empty()
        
        # Controles
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            start_button = st.button("🎥 INICIAR CÁMARA", use_container_width=True, type="primary")
        
        with col_btn2:
            stop_button = st.button("⏹️ DETENER", use_container_width=True)
        
        # Placeholder para mensajes
        message_placeholder = st.empty()
        
        # Inicializar estado de la cámara
        if 'camera_running' not in st.session_state:
            st.session_state['camera_running'] = False
        
        if start_button:
            st.session_state['camera_running'] = True
        
        if stop_button:
            st.session_state['camera_running'] = False
        
        # Loop de la cámara
        if st.session_state['camera_running']:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ No se puede acceder a la cámara")
                st.session_state['camera_running'] = False
            else:
                st.info("✅ Cámara activa - Los estudiantes pueden acercarse")
                
                frame_count = 0
                ultimo_reconocimiento = None
                
                while st.session_state['camera_running']:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("❌ Error al leer frame")
                        break
                    
                    # Detectar rostros
                    tiene_rostro, cantidad, rostros = detectar_rostro(frame)
                    
                    # Dibujar rectángulos
                    frame_display = frame.copy()
                    for (x, y, w, h) in rostros:
                        cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        cv2.putText(frame_display, "Rostro detectado", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Mostrar frame
                    frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame_rgb, channels="RGB")
                    
                    # Reconocer cada 30 frames (aprox. 1 segundo)
                    if tiene_rostro and cantidad == 1 and frame_count % 30 == 0:
                        message_placeholder.info("🔍 Reconociendo...")
                        
                        encontrado, estudiante, distancia, mensaje = reconocer_rostro_rapido(frame)
                        
                        if encontrado:
                            # Verificar cooldown
                            if puede_reconocer(estudiante['codigo']):
                                # Registrar asistencia
                                exito, msg = registrar_asistencia(
                                    estudiante['codigo'],
                                    estudiante['nombre'],
                                    estudiante['apellido'],
                                    distancia
                                )
                                
                                if exito:
                                    similitud = (1 - distancia) * 100
                                    message_placeholder.success(f"""
                                    ✅ **{estudiante['nombre']} {estudiante['apellido']}**  
                                    Código: {estudiante['codigo']} | Similitud: {similitud:.1f}%
                                    """)
                                    
                                    # Actualizar tabla en tiempo real
                                    if 'ultima_asistencia' not in st.session_state:
                                        st.session_state['ultima_asistencia'] = []
                                    
                                    st.session_state['ultima_asistencia'].append({
                                        'nombre': f"{estudiante['nombre']} {estudiante['apellido']}",
                                        'codigo': estudiante['codigo'],
                                        'hora': datetime.now().strftime('%H:%M:%S'),
                                        'similitud': f"{similitud:.1f}%"
                                    })
                                    
                                    time.sleep(1)  # Pausa para mostrar mensaje
                                else:
                                    message_placeholder.warning(f"⚠️ {msg}")
                            else:
                                message_placeholder.info("⏳ Espera unos segundos...")
                        else:
                            message_placeholder.warning("❌ Rostro no reconocido")
                    
                    frame_count += 1
                    
                    # Pequeña pausa para no saturar
                    time.sleep(0.033)  # ~30 FPS
                
                cap.release()
                message_placeholder.success("✅ Cámara detenida")
    
    with col2:
        st.subheader("📋 Asistencias Registradas")
        
        # Mostrar asistencias de hoy
        asistencias = obtener_asistencias_hoy()
        
        if len(asistencias) > 0:
            # Ordenar por hora descendente
            asistencias_sorted = sorted(asistencias, key=lambda x: x['hora'], reverse=True)
            
            # Mostrar las últimas 10
            for asist in asistencias_sorted[:10]:
                st.success(f"""
                **{asist['nombre']} {asist['apellido']}**  
                🆔 {asist['codigo']} | ⏰ {asist['hora']} | 📊 {asist['similitud']}
                """)
            
            st.markdown("---")
            st.metric("Total Hoy", len(asistencias))
        else:
            st.info("ℹ️ Aún no hay asistencias registradas")
        
        # Botón para refrescar
        if st.button("🔄 Actualizar", use_container_width=True):
            st.rerun()

# ============================================
# PÁGINA REPORTES
# ============================================
elif opcion == "📊 Reportes":
    st.header("Reportes y Consultas")
    
    tab1, tab2 = st.tabs(["📅 Asistencias Hoy", "👥 Estudiantes"])
    
    with tab1:
        st.subheader("Asistencias de Hoy")
        
        asistencias = obtener_asistencias_hoy()
        
        if len(asistencias) > 0:
            df = pd.DataFrame(asistencias)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Exportar CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar CSV",
                csv,
                f"asistencia_{date.today().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
            
            # Estadísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Asistencias", len(asistencias))
            with col2:
                similitudes = [float(a['similitud'].rstrip('%')) for a in asistencias]
                promedio = sum(similitudes) / len(similitudes)
                st.metric("Similitud Promedio", f"{promedio:.1f}%")
            with col3:
                total_est = len(cargar_estudiantes())
                if total_est > 0:
                    pct = (len(asistencias) / total_est) * 100
                    st.metric("% Asistencia", f"{pct:.1f}%")
        else:
            st.info("ℹ️ No hay asistencias hoy")
    
    with tab2:
        st.subheader("Lista de Estudiantes Registrados")
        
        estudiantes = cargar_estudiantes()
        
        if len(estudiantes) > 0:
            df_est = pd.DataFrame(estudiantes)
            df_est = df_est[['codigo', 'nombre', 'apellido', 'email', 'fecha_registro']]
            st.dataframe(df_est, use_container_width=True, hide_index=True)
            
            # Exportar CSV
            csv_est = df_est.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar Lista de Estudiantes",
                csv_est,
                "estudiantes.csv",
                "text/csv",
                use_container_width=True
            )
            
            st.metric("Total Estudiantes", len(estudiantes))
        else:
            st.info("ℹ️ No hay estudiantes registrados")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Sistema de Asistencia Tiempo Real v3.0 - DeepFace + OpenCV</p>
    <p>Optimizado para reconocimiento rápido en fila</p>
</div>
""", unsafe_allow_html=True)