import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from PIL import Image
import os
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Residuos IA",
    page_icon="♻️",
    layout="wide"
)

class WasteClassificationSystem:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = []
        self.img_size = (224, 224)
        self.original_size = (512, 384)
        self.data_path = "dataset"
        
    def discover_classes(self):
        """Descubre automáticamente las clases desde la estructura de carpetas"""
        try:
            if os.path.exists(self.data_path):
                self.class_names = [d for d in os.listdir(self.data_path) 
                                  if os.path.isdir(os.path.join(self.data_path, d))]
                self.class_names.sort()
                return True
            else:
                st.error(f"❌ No se encuentra la carpeta '{self.data_path}'")
                return False
        except Exception as e:
            st.error(f"Error descubriendo clases: {e}")
            return False
    
    def analyze_dataset(self):
        """Analiza el dataset y muestra estadísticas"""
        if not self.discover_classes():
            return None, 0
            
        stats = {}
        total_images = 0
        
        for class_name in self.class_names:
            class_path = os.path.join(self.data_path, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                stats[class_name] = len(images)
                total_images += len(images)
        
        return stats, total_images
    
    def create_data_generators(self, validation_split=0.2, batch_size=32):
        """Crea generadores de datos para entrenamiento y validación"""
        try:
            # Data augmentation para entrenamiento
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                validation_split=validation_split,
                fill_mode='nearest'
            )
            
            # Solo rescale para validación
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
            
            # Generador de entrenamiento
            train_generator = train_datagen.flow_from_directory(
                self.data_path,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            
            # Generador de validación
            val_generator = val_datagen.flow_from_directory(
                self.data_path,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
            
            self.class_names = list(train_generator.class_indices.keys())
            
            return train_generator, val_generator
            
        except Exception as e:
            st.error(f"Error creando generadores de datos: {e}")
            return None, None
    
    def create_model(self, num_classes, use_pretrained=True):
        """Crea el modelo de clasificación"""
        try:
            if use_pretrained:
                # Usar MobileNetV2 preentrenado
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet'
                )
                base_model.trainable = False
                
                model = models.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.Dropout(0.3),
                    layers.Dense(128, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.5),
                    layers.Dense(num_classes, activation='softmax')
                ])
            else:
                # Modelo CNN personalizado
                model = models.Sequential([
                    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                    layers.MaxPooling2D(2, 2),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D(2, 2),
                    layers.Conv2D(128, (3, 3), activation='relu'),
                    layers.MaxPooling2D(2, 2),
                    layers.Conv2D(128, (3, 3), activation='relu'),
                    layers.MaxPooling2D(2, 2),
                    layers.Flatten(),
                    layers.Dense(512, activation='relu'),
                    layers.Dropout(0.5),
                    layers.Dense(num_classes, activation='softmax')
                ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            st.error(f"Error creando modelo: {e}")
            return None
    
    def train_model(self, epochs=20, use_pretrained=True):
        """Entrena el modelo con los datos"""
        try:
            # Crear generadores de datos
            train_generator, val_generator = self.create_data_generators()
            
            if train_generator is None or val_generator is None:
                st.error("No se pudieron crear los generadores de datos")
                return None
            
            # Crear modelo
            num_classes = len(self.class_names)
            self.model = self.create_model(num_classes, use_pretrained)
            
            if self.model is None:
                st.error("No se pudo crear el modelo")
                return None
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=5, 
                    restore_best_weights=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.2, 
                    patience=3,
                    monitor='val_loss'
                )
            ]
            
            # Entrenar
            st.info(f"🚀 Entrenando modelo con {num_classes} clases...")
            st.info(f"Clases: {', '.join(self.class_names)}")
            
            self.history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            return self.history
            
        except Exception as e:
            st.error(f"Error entrenando modelo: {e}")
            return None
    
    def evaluate_model(self):
        """Evalúa el modelo entrenado"""
        if self.model is None:
            st.error("No hay modelo entrenado para evaluar")
            return None
            
        try:
            # Crear generador de validación
            _, val_generator = self.create_data_generators()
            
            if val_generator is None:
                st.error("No se pudo crear el generador de validación")
                return None
            
            # Evaluar
            evaluation = self.model.evaluate(val_generator, verbose=0)
            
            # Predecir para matriz de confusión
            val_generator.reset()
            predictions = self.model.predict(val_generator, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = val_generator.classes
            
            return evaluation, true_classes, predicted_classes
            
        except Exception as e:
            st.error(f"Error evaluando modelo: {e}")
            return None

    def predict_image(self, image):
        """Predice la clase de una imagen"""
        if self.model is None:
            st.error("El modelo no ha sido entrenado")
            return None, 0.0
            
        try:
            # Preprocesar imagen
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convertir a RGB si es necesario
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Redimensionar y normalizar
            image = cv2.resize(image, self.img_size)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Predecir
            predictions = self.model.predict(image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            predicted_class = self.class_names[predicted_class_idx]
            
            return predicted_class, confidence
            
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            return None, 0.0
    
    def save_model(self, path="waste_classifier_model.h5"):
        """Guarda el modelo entrenado"""
        if self.model is not None:
            self.model.save(path)
            return True
        return False
    
    def load_model(self, path="waste_classifier_model.h5"):
        """Carga un modelo previamente guardado"""
        try:
            self.model = tf.keras.models.load_model(path)
            # Si no tenemos class_names, intentar descubrirlas
            if not self.class_names:
                self.discover_classes()
            return True
        except:
            return False

def plot_training_history(history):
    """Grafica el historial de entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Precisión
    ax1.plot(history.history['accuracy'], label='Precisión Entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Precisión Validación')
    ax1.set_title('Precisión del Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Precisión')
    ax1.legend()
    ax1.grid(True)
    
    # Pérdida
    ax2.plot(history.history['loss'], label='Pérdida Entrenamiento')
    ax2.plot(history.history['val_loss'], label='Pérdida Validación')
    ax2.set_title('Pérdida del Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(true_classes, pred_classes, class_names):
    """Grafica la matriz de confusión"""
    cm = confusion_matrix(true_classes, pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    return plt.gcf()

def main():
    st.title("♻️ Sistema de Clasificación de Residuos con IA")
    st.markdown("---")
    
    # Inicializar sistema
    system = WasteClassificationSystem()
    
    # Sidebar
    st.sidebar.title("Navegación")
    app_mode = st.sidebar.selectbox(
        "Selecciona el modo",
        ["Análisis de Datos", "Entrenamiento", "Clasificación", "Evaluación"]
    )
    
    if app_mode == "Análisis de Datos":
        show_data_analysis(system)
    elif app_mode == "Entrenamiento":
        show_training_page(system)
    elif app_mode == "Clasificación":
        show_classification_page(system)
    elif app_mode == "Evaluación":
        show_evaluation_page(system)

def show_data_analysis(system):
    """Muestra el análisis del dataset"""
    st.header("📊 Análisis del Dataset")
    
    if not system.discover_classes():
        st.error("""
        No se pudo encontrar la estructura del dataset.
        
        Asegúrate de que tienes la siguiente estructura:
        ```
        dataset/
        ├── clase1/
        │   ├── imagen1.jpg
        │   ├── imagen2.jpg
        │   └── ...
        ├── clase2/
        │   ├── imagen1.jpg
        │   └── ...
        └── ...
        ```
        """)
        return
    
    # Estadísticas del dataset
    stats, total_images = system.analyze_dataset()
    
    if stats:
        st.subheader("Estadísticas del Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Clases", len(system.class_names))
        with col2:
            st.metric("Total de Imágenes", total_images)
        with col3:
            avg_per_class = total_images // len(system.class_names)
            st.metric("Promedio por Clase", avg_per_class)
        
        # Gráfico de distribución
        st.subheader("Distribución por Clase")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = list(stats.keys())
        counts = list(stats.values())
        
        bars = ax.bar(classes, counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3'])
        ax.set_title('Distribución de Imágenes por Clase')
        ax.set_xlabel('Clases')
        ax.set_ylabel('Número de Imágenes')
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{count}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Mostrar información detallada
        st.subheader("Detalles por Clase")
        detail_df = pd.DataFrame({
            'Clase': list(stats.keys()),
            'Imágenes': list(stats.values()),
            'Porcentaje': [f"{(count/total_images)*100:.1f}%" for count in stats.values()]
        })
        st.dataframe(detail_df)

def show_training_page(system):
    """Muestra la página de entrenamiento"""
    st.header("🔧 Entrenamiento del Modelo")
    
    if not system.discover_classes():
        st.error("Primero necesitas configurar correctamente el dataset")
        return
    
    st.info(f"🔄 Se detectaron {len(system.class_names)} clases: {', '.join(system.class_names)}")
    
    # Configuración del entrenamiento
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuración del Modelo")
        use_pretrained = st.checkbox("Usar modelo preentrenado (MobileNetV2)", value=True)
        epochs = st.slider("Número de épocas", min_value=5, max_value=50, value=20)
        batch_size = st.slider("Tamaño del batch", min_value=16, max_value=64, value=32)
        validation_split = st.slider("Porcentaje validación", min_value=0.1, max_value=0.3, value=0.2)
    
    with col2:
        st.subheader("Información del Dataset")
        stats, total_images = system.analyze_dataset()
        if stats:
            st.write(f"**Total de imágenes:** {total_images}")
            st.write(f"**Imágenes de entrenamiento:** {int(total_images * (1-validation_split))}")
            st.write(f"**Imágenes de validación:** {int(total_images * validation_split)}")
            st.write(f"**Tamaño de imagen:** {system.img_size}")
    
    # Botón de entrenamiento
    if st.button("🚀 Iniciar Entrenamiento", type="primary"):
        if total_images == 0:
            st.error("No hay imágenes para entrenar. Verifica la estructura del dataset.")
            return
            
        with st.spinner("Entrenando modelo... Esto puede tomar varios minutos."):
            # Entrenar modelo
            history = system.train_model(
                epochs=epochs,
                use_pretrained=use_pretrained
            )
            
            if history is not None:
                st.success("✅ Entrenamiento completado exitosamente!")
                
                # Mostrar gráficos de entrenamiento
                fig = plot_training_history(history)
                st.pyplot(fig)
                
                # Guardar modelo
                if system.save_model():
                    st.success("💾 Modelo guardado como 'waste_classifier_model.h5'")
                
                # Mostrar métricas finales
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Precisión Final (Entrenamiento)", f"{final_train_acc:.2%}")
                with col2:
                    st.metric("Precisión Final (Validación)", f"{final_val_acc:.2%}")

def show_classification_page(system):
    """Muestra la página de clasificación"""
    st.header("🔍 Clasificación de Residuos")
    
    # Intentar cargar modelo existente
    model_loaded = system.load_model()
    if not model_loaded and system.model is None:
        st.warning("⚠️ No hay modelo entrenado. Entrena un modelo primero.")
        return
    
    st.success("✅ Modelo cargado correctamente")
    
    # Opciones de entrada
    input_method = st.radio(
        "Selecciona el método de entrada:",
        ["Subir imagen", "Probar con imágenes del dataset"]
    )
    
    image = None
    image_source = ""
    
    if input_method == "Subir imagen":
        uploaded_file = st.file_uploader("Sube una imagen de residuo", 
                                       type=['jpg', 'jpeg', '.png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_source = "uploaded"
    
    else:  # Probar con imágenes del dataset
        if system.discover_classes() and system.class_names:
            selected_class = st.selectbox("Selecciona una clase:", system.class_names)
            if selected_class:
                class_path = os.path.join(system.data_path, selected_class)
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    selected_image = st.selectbox("Selecciona una imagen:", images)
                    if selected_image:
                        image_path = os.path.join(class_path, selected_image)
                        image = Image.open(image_path)
                        image_source = f"dataset - Clase real: {selected_class}"
                        st.image(image, caption=f"Imagen de prueba: {selected_image}", use_column_width=True)
    
    # Clasificar si hay imagen
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen de Entrada")
            st.image(image, use_column_width=True)
            
            if image_source:
                st.write(f"**Fuente:** {image_source}")
        
        with col2:
            st.subheader("Resultado de la Clasificación")
            
            # Predicción real
            predicted_class, confidence = system.predict_image(image)
            
            if predicted_class:
                # Mostrar resultado
                confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                emoji = "✅" if confidence > 0.8 else "⚠️" if confidence > 0.6 else "❌"
                
                st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6; border-left: 5px solid {confidence_color};'>
                    <h3 style='color: {confidence_color}; margin-top: 0;'>{emoji} {predicted_class.upper()}</h3>
                    <h4 style='color: {confidence_color};'>Confianza: {confidence:.2%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Barra de progreso
                st.progress(float(confidence))
                
                # Información de reciclaje
                recycling_info = get_recycling_info(predicted_class)
                st.info(recycling_info)
                
                # Verificar si conocemos la clase real (para modo dataset)
                if "Clase real:" in image_source:
                    real_class = image_source.split("Clase real: ")[1]
                    if real_class.lower() == predicted_class.lower():
                        st.success("🎯 ¡Clasificación correcta!")
                    else:
                        st.error(f"❌ Clasificación incorrecta. Debería ser: {real_class}")

def show_evaluation_page(system):
    """Muestra la página de evaluación"""
    st.header("📊 Evaluación del Modelo")
    
    # Verificar si hay modelo entrenado
    if system.model is None:
        if not system.load_model():
            st.error("No hay modelo entrenado para evaluar. Entrena un modelo primero.")
            return
    
    if st.button("🔍 Ejecutar Evaluación Completa"):
        with st.spinner("Evaluando modelo..."):
            results = system.evaluate_model()
            
            if results:
                evaluation, true_classes, pred_classes = results
                
                # Métricas principales
                st.subheader("Métricas Principales")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Pérdida", f"{evaluation[0]:.4f}")
                with col2:
                    st.metric("Precisión", f"{evaluation[1]:.2%}")
                with col3:
                    # Calcular precisión manualmente para reporte
                    accuracy = np.sum(true_classes == pred_classes) / len(true_classes)
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col4:
                    # Tiempo de inferencia estimado
                    st.metric("Clases", len(system.class_names))
                
                # Matriz de confusión
                st.subheader("Matriz de Confusión")
                fig = plot_confusion_matrix(true_classes, pred_classes, system.class_names)
                st.pyplot(fig)
                
                # Reporte de clasificación
                st.subheader("Reporte de Clasificación Detallado")
                report = classification_report(true_classes, pred_classes, 
                                            target_names=system.class_names,
                                            output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))
                
                # Gráfico de precisión por clase
                st.subheader("Precisión por Clase")
                class_accuracy = []
                for i, class_name in enumerate(system.class_names):
                    class_mask = true_classes == i
                    if np.sum(class_mask) > 0:
                        class_acc = np.sum(pred_classes[class_mask] == i) / np.sum(class_mask)
                        class_accuracy.append(class_acc)
                    else:
                        class_accuracy.append(0)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(system.class_names, class_accuracy, color='skyblue')
                ax.set_title('Precisión por Clase')
                ax.set_ylabel('Precisión')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                
                # Añadir valores en las barras
                for bar, acc in zip(bars, class_accuracy):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.2%}', ha='center', va='bottom')
                
                st.pyplot(fig)

def get_recycling_info(waste_type):
    """Devuelve información de reciclaje para el tipo de residuo"""
    info = {
        'cardboard': "📦 **Cartón** - Contenedor AZUL. Aplastar antes de reciclar.",
        'glass': "♻️ **Vidrio** - Contenedor VERDE. Lavar antes de reciclar.",
        'metal': "🥫 **Metal** - Contenedor AMARILLO. Enjuagar latas.",
        'paper': "📄 **Papel** - Contenedor AZUL. Sin grapas o plásticos.",
        'plastic': "🥤 **Plástico** - Contenedor AMARILLO. Limpiar y compactar.",
        'trash': "🗑️ **Residuos** - Contenedor GRIS/MARRÓN. No reciclable."
    }
    return info.get(waste_type.lower(), "ℹ️ Consulta las normas locales de reciclaje.")

if __name__ == "__main__":
    main()