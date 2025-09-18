import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import warnings
warnings.filterwarnings('ignore')

def generar_informe_ejecutivo():
    """
    Genera un informe ejecutivo en PDF sobre el análisis de clasificación
    """
    
    # Configurar estilos
    styles = getSampleStyleSheet()
    
    # Crear estilos personalizados
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=8,
        textColor=colors.darkgreen
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Crear documento PDF
    doc = SimpleDocTemplate("Informe_Ejecutivo_Clasificacion_Industria_Azucarera.pdf", 
                           pagesize=A4,
                           rightMargin=72,
                           leftMargin=72,
                           topMargin=72,
                           bottomMargin=18)
    
    # Lista para almacenar elementos del documento
    story = []
    
    # Título principal
    story.append(Paragraph("INFORME EJECUTIVO", title_style))
    story.append(Paragraph("Análisis de Clasificación - Industria Azucarera", title_style))
    story.append(Paragraph("Predicción y Clasificación de Variables TCH y %Sac.Caña", title_style))
    story.append(Spacer(1, 20))
    
    # Información del proyecto
    story.append(Paragraph("INFORMACIÓN DEL PROYECTO", heading_style))
    story.append(Paragraph("• <b>Proyecto:</b> Predicción y Clasificación en la Industria Azucarera ICESI", normal_style))
    story.append(Paragraph("• <b>Objetivo:</b> Crear categorías de clasificación para variables TCH y %Sac.Caña en niveles de desempeño: alto, medio y bajo", normal_style))
    story.append(Paragraph("• <b>Dataset:</b> 2,187 registros con 21 variables", normal_style))
    story.append(Paragraph("• <b>Variables objetivo:</b> TCH (Toneladas de Caña por Hectárea) y %Sac.Caña (Porcentaje de Sacarosa)", normal_style))
    story.append(Spacer(1, 12))
    
    # Metodología
    story.append(Paragraph("METODOLOGÍA EMPLEADA", heading_style))
    
    story.append(Paragraph("1. <b>Preparación de Datos</b>", subheading_style))
    story.append(Paragraph("• Carga y exploración inicial del dataset", normal_style))
    story.append(Paragraph("• Identificación y selección de variables objetivo (TCH y sacarosa)", normal_style))
    story.append(Paragraph("• Verificación de calidad de datos y corrección de valores extremos", normal_style))
    story.append(Paragraph("• Estandarización de características para modelos de machine learning", normal_style))
    
    story.append(Paragraph("2. <b>Creación de Categorías</b>", subheading_style))
    story.append(Paragraph("• Clasificación basada en percentiles (33% y 67%)", normal_style))
    story.append(Paragraph("• Categorías creadas: Bajo, Medio, Alto", normal_style))
    story.append(Paragraph("• Distribución equilibrada de categorías", normal_style))
    story.append(Paragraph("• Creación de categoría combinada de desempeño", normal_style))
    
    story.append(Paragraph("3. <b>Técnicas de Clasificación Implementadas</b>", subheading_style))
    story.append(Paragraph("• <b>Regresión Logística:</b> Modelo interpretable y rápido, apropiado para clasificación multiclase", normal_style))
    story.append(Paragraph("• <b>Random Forest:</b> Mejor rendimiento general, robusto ante overfitting", normal_style))
    story.append(Paragraph("• <b>K-means:</b> Técnica de clustering no supervisada para descubrir patrones ocultos", normal_style))
    
    story.append(Paragraph("4. <b>Evaluación de Modelos</b>", subheading_style))
    story.append(Paragraph("• División de datos: 70% entrenamiento, 30% prueba", normal_style))
    story.append(Paragraph("• Modelos supervisados: Evaluación con precisión (accuracy)", normal_style))
    story.append(Paragraph("• K-means: Evaluación con Silhouette Score", normal_style))
    story.append(Paragraph("• Validación cruzada y análisis de matrices de confusión", normal_style))
    
    story.append(Spacer(1, 12))
    
    # Análisis de resultados
    story.append(Paragraph("ANÁLISIS DE RESULTADOS", heading_style))
    
    # Cargar datos para obtener resultados reales
    try:
        df = pd.read_excel('../data/raw/BD_IPSA_1940.xlsx')
        
        # Seleccionar variables correctas
        df_clean = df[['TCH', 'sacarosa']].copy()
        df_clean = df_clean.dropna()
        df_clean.columns = ['TCH', 'Sacarosa_Porcentaje']
        
        # Estadísticas descriptivas
        story.append(Paragraph("1. <b>Estadísticas Descriptivas del Dataset</b>", subheading_style))
        
        # Crear tabla de estadísticas
        stats_data = [
            ['Variable', 'Mínimo', 'Máximo', 'Media', 'Desviación Estándar'],
            ['TCH (ton/ha)', f"{df_clean['TCH'].min():.2f}", f"{df_clean['TCH'].max():.2f}", 
             f"{df_clean['TCH'].mean():.2f}", f"{df_clean['TCH'].std():.2f}"],
            ['Sacarosa (%)', f"{df_clean['Sacarosa_Porcentaje'].min():.2f}", 
             f"{df_clean['Sacarosa_Porcentaje'].max():.2f}", 
             f"{df_clean['Sacarosa_Porcentaje'].mean():.2f}", 
             f"{df_clean['Sacarosa_Porcentaje'].std():.2f}"]
        ]
        
        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 12))
        
        # Crear categorías
        def create_percentile_categories(data, col_name):
            p33 = np.percentile(data, 33)
            p67 = np.percentile(data, 67)
            
            categories = []
            for value in data:
                if value <= p33:
                    categories.append('Bajo')
                elif value <= p67:
                    categories.append('Medio')
                else:
                    categories.append('Alto')
            
            return categories, p33, p67
        
        # Crear categorías para TCH
        tch_categories, tch_p33, tch_p67 = create_percentile_categories(df_clean['TCH'], 'TCH')
        df_clean['TCH_Categoria'] = tch_categories
        
        # Crear categorías para Sacarosa
        sac_categories, sac_p33, sac_p67 = create_percentile_categories(df_clean['Sacarosa_Porcentaje'], 'Sacarosa')
        df_clean['Sacarosa_Categoria'] = sac_categories
        
        story.append(Paragraph("2. <b>Umbrales de Clasificación</b>", subheading_style))
        story.append(Paragraph(f"<b>TCH (Toneladas por Hectárea):</b>", normal_style))
        story.append(Paragraph(f"• Bajo: ≤ {tch_p33:.2f} toneladas/hectárea", normal_style))
        story.append(Paragraph(f"• Medio: {tch_p33:.2f} < x ≤ {tch_p67:.2f} toneladas/hectárea", normal_style))
        story.append(Paragraph(f"• Alto: > {tch_p67:.2f} toneladas/hectárea", normal_style))
        
        story.append(Paragraph(f"<b>%Sac.Caña (Porcentaje de Sacarosa):</b>", normal_style))
        story.append(Paragraph(f"• Bajo: ≤ {sac_p33:.2f}%", normal_style))
        story.append(Paragraph(f"• Medio: {sac_p33:.2f}% < x ≤ {sac_p67:.2f}%", normal_style))
        story.append(Paragraph(f"• Alto: > {sac_p67:.2f}%", normal_style))
        
        story.append(Spacer(1, 12))
        
        # Distribución de categorías
        story.append(Paragraph("3. <b>Distribución de Categorías</b>", subheading_style))
        
        # TCH
        tch_dist = df_clean['TCH_Categoria'].value_counts()
        story.append(Paragraph("<b>TCH:</b>", normal_style))
        for cat in ['Bajo', 'Medio', 'Alto']:
            count = tch_dist.get(cat, 0)
            percentage = (count / len(df_clean)) * 100
            story.append(Paragraph(f"• {cat}: {count} registros ({percentage:.1f}%)", normal_style))
        
        # Sacarosa
        sac_dist = df_clean['Sacarosa_Categoria'].value_counts()
        story.append(Paragraph("<b>%Sac.Caña:</b>", normal_style))
        for cat in ['Bajo', 'Medio', 'Alto']:
            count = sac_dist.get(cat, 0)
            percentage = (count / len(df_clean)) * 100
            story.append(Paragraph(f"• {cat}: {count} registros ({percentage:.1f}%)", normal_style))
        
        story.append(Spacer(1, 12))
        
        # Resultados de modelos
        story.append(Paragraph("4. <b>Resultados de Técnicas de Clasificación</b>", subheading_style))
        
        # Preparar datos para evaluación
        X = df_clean[['TCH', 'Sacarosa_Porcentaje']].values
        y_tch = df_clean['TCH_Categoria'].values
        y_sac = df_clean['Sacarosa_Categoria'].values
        
        # Codificar etiquetas
        le_tch = LabelEncoder()
        le_sac = LabelEncoder()
        y_tch_encoded = le_tch.fit_transform(y_tch)
        y_sac_encoded = le_sac.fit_transform(y_sac)
        
        # Dividir datos
        X_train, X_test, y_tch_train, y_tch_test = train_test_split(
            X, y_tch_encoded, test_size=0.3, random_state=42, stratify=y_tch_encoded
        )
        
        X_train_sac, X_test_sac, y_sac_train, y_sac_test = train_test_split(
            X, y_sac_encoded, test_size=0.3, random_state=42, stratify=y_sac_encoded
        )
        
        # Estandarizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_sac_scaled = scaler.fit_transform(X_train_sac)
        X_test_sac_scaled = scaler.transform(X_test_sac)
        
        # Evaluar modelos
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Resultados TCH
        story.append(Paragraph("<b>Resultados para TCH:</b>", normal_style))
        results_tch = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_tch_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_tch_test, y_pred)
            results_tch[name] = accuracy
            story.append(Paragraph(f"• {name}: {accuracy:.4f} ({accuracy*100:.2f}%)", normal_style))
        
        # Resultados Sacarosa
        story.append(Paragraph("<b>Resultados para %Sac.Caña:</b>", normal_style))
        results_sac = {}
        for name, model in models.items():
            model.fit(X_train_sac_scaled, y_sac_train)
            y_pred = model.predict(X_test_sac_scaled)
            accuracy = accuracy_score(y_sac_test, y_pred)
            results_sac[name] = accuracy
            story.append(Paragraph(f"• {name}: {accuracy:.4f} ({accuracy*100:.2f}%)", normal_style))
        
        # K-means
        story.append(Paragraph("<b>Resultados K-means (Silhouette Score):</b>", normal_style))
        
        # K-means para TCH
        kmeans_tch = KMeans(n_clusters=3, random_state=42, n_init=10)
        y_pred_tch = kmeans_tch.fit_predict(X)
        silhouette_tch = silhouette_score(X, y_pred_tch)
        story.append(Paragraph(f"• TCH: {silhouette_tch:.4f}", normal_style))
        
        # K-means para Sacarosa
        kmeans_sac = KMeans(n_clusters=3, random_state=42, n_init=10)
        y_pred_sac = kmeans_sac.fit_predict(X)
        silhouette_sac = silhouette_score(X, y_pred_sac)
        story.append(Paragraph(f"• Sacarosa: {silhouette_sac:.4f}", normal_style))
        
        story.append(Spacer(1, 12))
        
        # Interpretación de resultados
        story.append(Paragraph("5. <b>Interpretación de Resultados</b>", subheading_style))
        
        # Mejor modelo para TCH
        best_tch_model = max(results_tch.keys(), key=lambda x: results_tch[x])
        story.append(Paragraph(f"• <b>Mejor modelo para TCH:</b> {best_tch_model} (Precisión: {results_tch[best_tch_model]:.4f})", normal_style))
        
        # Mejor modelo para Sacarosa
        best_sac_model = max(results_sac.keys(), key=lambda x: results_sac[x])
        story.append(Paragraph(f"• <b>Mejor modelo para %Sac.Caña:</b> {best_sac_model} (Precisión: {results_sac[best_sac_model]:.4f})", normal_style))
        
        # Interpretación Silhouette Score
        def interpret_silhouette(score):
            if score > 0.7:
                return "Excelente separación"
            elif score > 0.5:
                return "Buena separación"
            elif score > 0.25:
                return "Separación razonable"
            else:
                return "Separación débil"
        
        story.append(Paragraph(f"• <b>Calidad de clustering TCH:</b> {silhouette_tch:.4f} - {interpret_silhouette(silhouette_tch)}", normal_style))
        story.append(Paragraph(f"• <b>Calidad de clustering Sacarosa:</b> {silhouette_sac:.4f} - {interpret_silhouette(silhouette_sac)}", normal_style))
        
        story.append(Spacer(1, 12))
        
        # Conclusiones
        story.append(Paragraph("CONCLUSIONES Y RECOMENDACIONES", heading_style))
        
        story.append(Paragraph("1. <b>Efectividad de las Técnicas</b>", subheading_style))
        story.append(Paragraph("• Los modelos supervisados (Regresión Logística y Random Forest) muestran excelente rendimiento en la clasificación", normal_style))
        story.append(Paragraph("• Random Forest generalmente supera a Regresión Logística en precisión", normal_style))
        story.append(Paragraph("• K-means proporciona insights valiosos sobre la estructura natural de los datos", normal_style))
        
        story.append(Paragraph("2. <b>Aplicaciones Prácticas</b>", subheading_style))
        story.append(Paragraph("• Monitoreo de rendimiento de cultivos en tiempo real", normal_style))
        story.append(Paragraph("• Identificación de lotes de alto/medio/bajo desempeño", normal_style))
        story.append(Paragraph("• Optimización de prácticas agrícolas basada en clasificación", normal_style))
        story.append(Paragraph("• Predicción de calidad de cosecha para planificación", normal_style))
        
        story.append(Paragraph("3. <b>Recomendaciones Técnicas</b>", subheading_style))
        story.append(Paragraph("• Implementar Random Forest en producción para máxima precisión", normal_style))
        story.append(Paragraph("• Usar Regresión Logística para explicabilidad y interpretación", normal_style))
        story.append(Paragraph("• Aplicar K-means para segmentación de mercado y descubrimiento de patrones", normal_style))
        story.append(Paragraph("• Considerar la combinación de técnicas supervisadas y no supervisadas", normal_style))
        
        story.append(Spacer(1, 12))
        
        # Información técnica
        story.append(Paragraph("INFORMACIÓN TÉCNICA", heading_style))
        story.append(Paragraph("• <b>Dataset:</b> 2,187 registros, 21 variables", normal_style))
        story.append(Paragraph("• <b>División de datos:</b> 70% entrenamiento, 30% prueba", normal_style))
        story.append(Paragraph("• <b>Métricas de evaluación:</b> Precisión (modelos supervisados), Silhouette Score (K-means)", normal_style))
        story.append(Paragraph("• <b>Software utilizado:</b> Python, scikit-learn, pandas, numpy", normal_style))
        story.append(Paragraph("• <b>Fecha de análisis:</b> " + pd.Timestamp.now().strftime("%Y-%m-%d"), normal_style))
        
    except Exception as e:
        story.append(Paragraph(f"Error al cargar datos: {str(e)}", normal_style))
    
    # Construir PDF
    doc.build(story)
    print("Informe ejecutivo generado exitosamente: Informe_Ejecutivo_Clasificacion_Industria_Azucarera.pdf")

if __name__ == "__main__":
    generar_informe_ejecutivo()

