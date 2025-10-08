# ♻️ Sistema de Clasificación de Residuos con IA

Desarrollado por Erick Díaz C.I.29963164

Materia: Inteligencia Artificial

Periodo: 2025C

## Descripción
Este proyecto implementa un sistema inteligente de clasificación de residuos utilizando técnicas avanzadas de 
Inteligencia Artificial y Visión Artificial. El sistema es capaz de identificar y clasificar diferentes tipos 
de residuos a partir de imágenes, facilitando el proceso de reciclaje y contribuyendo a la gestión sostenible de desechos.

## Objetivos Principales

- Clasificación Automática: Identificar automáticamente el tipo de residuo en imágenes
- Interfaz Amigable: Proporcionar una interfaz web intuitiva para la clasificación
- Entrenamiento Flexible: Permitir el entrenamiento de modelos personalizados
- Evaluación Completa: Ofrecer métricas detalladas del rendimiento del modelo

## Tecnologías
- **Python 3.8+**
- **TensorFlow/Keras** - Modelos de deep learning
- **Streamlit** - Interfaz web
- **OpenCV** - Procesamiento de imágenes
- **Scikit-learn** - Métricas de evaluación

## Estructura del Proyecto

Proyecto1/

├── main.py # Aplicación principal

├── waste_classifier_model.h5 # Modelo entrenado

├── requirements.txt # Dependencias

├── dataset/ # Dataset de entrenamiento

│ ├── cardboard/

│ ├── glass/

│ ├── metal/

│ ├── paper/

│ ├── plastic/

│ └── trash/

├── test

## Pasos de Instalación:

1. Clonar o descargar el proyecto
   git clone https://github.com/ErickMDR/Proyecto1_IA
   cd Proyecto1

2. Instalar dependencias
   pip install -r requirements.txt

3. Agregar imágenes de entrenamiento
   - Coloca las imágenes en las carpetas correspondientes
   - Formatos soportados: JPG, JPEG, PNG
   - Tamaño recomendado: mínimo 224x224 píxeles

## Uso de la Aplicación

Ejecutar con:
streamlit run app.py

La aplicación se abrirá en tu navegador en http://localhost:8501

## Módulos Principales:

1. ANÁLISIS DE DATOS
   - Visualización de la distribución del dataset
   - Estadísticas por clase de residuo
   - Información detallada de las imágenes disponibles

2. ENTRENAMIENTO DEL MODELO

   CONFIGURACIÓN DEL MODELO:
   - Usar modelo preentrenado (MobileNetV2) o CNN personalizado
   - Ajustar número de épocas (5-50)
   - Configurar tamaño del batch (16-64)
   - Definir porcentaje de validación (10%-30%)
   - Controlar pasos por época (automático o manual)

   PROCESO DE ENTRENAMIENTO:
   - Entrenamiento con data augmentation
   - Early stopping para prevenir overfitting
   - Reducción de learning rate dinámica
   - Guardado automático del modelo

4. CLASIFICACIÓN
   
   MÉTODOS DE ENTRADA:
   - Subir imagen desde dispositivo
   - Probar con imágenes del dataset

   RESULTADOS:
   - Clase predicha con porcentaje de confianza
   - Información de reciclaje específica
   - Validación contra clase real (cuando está disponible)

6. EVALUACIÓN
   
   MÉTRICAS:
   - Precisión y pérdida del modelo
   - Matriz de confusión
   - Reporte de clasificación detallado
   - Precisión por clase individual

## ARQUITECTURA DEL MODELO

OPCIONES DE MODELO:

1. MobileNetV2 con Transfer Learning (Recomendado)
   - Base preentrenada en ImageNet
   - Capas fully connected personalizadas
   - Dropout para regularización
   - Batch normalization

2. CNN Personalizado
   - 4 capas convolucionales
   - Max pooling
   - Capas fully connected
   - Dropout para prevenir overfitting

PREPROCESAMIENTO DE DATOS:
- Redimensionamiento: 224x224 píxeles
- Normalización: Valores entre 0-1
- Data Augmentation:
  - Rotación (±20°)
  - Desplazamiento horizontal/vertical
  - Volteo horizontal
  - Zoom aleatorio

## MÉTRICAS DE EVALUACIÓN

El sistema proporciona múltiples métricas:
- Precisión General: Accuracy del modelo
- Matriz de Confusión: Visualización de aciertos/errores
- Precisión por Clase: Rendimiento individual por tipo de residuo
- Pérdida: Función de pérdida durante entrenamiento/validación
- F1-Score: Balance entre precisión y recall

REQUISITOS DEL DATASET:
- Mínimo recomendado: 100 imágenes por clase
- Formatos: JPG, JPEG, PNG
- Variedad en ángulos, iluminación y fondos
- Imágenes claras y bien enfocadas

## CONFIGURACIÓN AVANZADA

HIPERPARÁMETROS AJUSTABLES:

En la interfaz de entrenamiento puedes modificar:
- Épocas: 5-50
- Batch Size: 16-64
- Validation Split: 0.1-0.3
- Steps per Epoch: Personalizable
- Model Architecture: MobileNetV2 o CNN personalizado

TIPOS DE RESIDUOS SOPORTADOS:

1. ♻️ RECICLABLES
   - cardboard: Cartón y papel
   - glass: Vidrio
   - metal: Metales
   - paper: Papel
   - plastic: Plásticos

2. 🗑️ NO RECICLABLES
   - trash: Residuos generales

## DESPLIEGUE

DESPLIEGUE LOCAL:
streamlit run app.py
