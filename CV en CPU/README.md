# AnalyticVision

Sistema de detección y clasificación de objetos en imágenes basado en representación analítica de formas. No requiere datos de entrenamiento, funciona en CPU estándar y permite añadir nuevas clases en tiempo real sin reentrenar.

---

## Qué hace

La mayoría de los sistemas de visión por computadora modernos necesitan miles de imágenes etiquetadas, una GPU potente y días de entrenamiento para reconocer objetos. AnalyticVision funciona de forma diferente: describe matemáticamente la forma de cada objeto mediante expresiones analíticas y usa esa descripción para comparar y clasificar.

El resultado es un sistema que aprende a reconocer una clase nueva a partir de **una sola imagen de referencia**, tarda **menos de 100ms por imagen en CPU** y puede extenderse con nuevas clases sin tocar el código ni el modelo.

---

## Cómo funciona

El pipeline tiene cinco módulos encadenados:

```
Imagen de entrada
      │
      ▼
graph_extractor      Extrae los puntos de una curva en una gráfica
      │
      ▼
analytical_engine    Ajusta una expresión matemática sobre esos puntos
      │              f(x) = a·sin(b·x + c) + d   [R² = 0.998]
      ▼
shape_encoder        Describe la forma de un objeto como vector analítico
      │              usando Descriptores de Fourier Elípticos + geometría
      ▼
detector_classifier  Detecta regiones en la imagen y clasifica cada una
      │              comparando su vector con la biblioteca de referencias
      ▼
score_calibrator     Convierte la similitud en una confianza calibrada
                     con expresión analítica explícita f(score) → precisión
```

### El ciclo de uso

```
OFFLINE (una vez por clase, sin datos de entrenamiento)
────────────────────────────────────────────────────────
sistema.registrar_clase("letra_A", "referencia_A.png")
sistema.registrar_clase("circulo", "referencia_circulo.png")
sistema.guardar("mi_biblioteca.npz")

ONLINE (tiempo real)
─────────────────────
sistema.cargar("mi_biblioteca.npz")
detecciones = sistema.detectar("documento.png")
sistema.visualizar("documento.png", detecciones)

EXTENSIÓN (sin reentrenar, en cualquier momento)
─────────────────────────────────────────────────
sistema.registrar_clase("nueva_clase", "una_imagen.png")
```

---

## Instalación

```bash
pip install opencv-python numpy scipy scikit-learn matplotlib
```

OCR opcional para leer números de ejes automáticamente:
```bash
pip install pytesseract pillow
# + instalar Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
```

TensorFlow opcional para la capa de calibración en Keras:
```bash
pip install tensorflow
```

---

## Archivos del proyecto

| Archivo | Función |
|---|---|
| `graph_extractor.py` | Extrae puntos `(x, y)` de una gráfica en imagen |
| `analytical_engine.py` | Ajusta la expresión analítica que mejor describe esos puntos |
| `shape_encoder.py` | Codifica una forma como vector de features analíticos |
| `detector_classifier.py` | Detecta y clasifica objetos en imágenes en tiempo real |
| `score_calibrator.py` | Calibra los scores de un clasificador Keras con fórmula explícita |

---

## Posibles aplicaciones

### Digitalización de documentos y gráficas científicas

Extracción automática de datos de gráficas publicadas en papers, informes o libros de texto. En lugar de leer manualmente los valores de una curva, el sistema produce la lista de puntos y la expresión matemática que la describe, lista para usar en cálculos.

```python
from graph_extractor import extraer_puntos
from analytical_engine import analizar

puntos    = extraer_puntos("figura_3.png", x_min=0, x_max=10)
resultado = analizar(puntos)
# → f(x) = 2.1·e^(-0.3·x)·sin(1.57·x) + 0.5   R²=0.994
```

---

### Reconocimiento de escritura y símbolos sin dataset

Sistemas de OCR o reconocimiento de símbolos técnicos (matemáticos, químicos, eléctricos, musicales) donde no existen datasets etiquetados o el dominio es demasiado específico para modelos genéricos. Una imagen de referencia por símbolo es suficiente.

**Ejemplos concretos:**
- Lectura de matrículas en formatos no estándar
- Reconocimiento de marcas o sellos en documentos históricos
- Identificación de símbolos en planos técnicos o esquemas eléctricos
- Reconocimiento de notación musical manuscrita
- Lectura de ecuaciones matemáticas escritas a mano

---

### Inspección industrial en línea de producción

Detección de defectos, piezas mal colocadas o elementos faltantes en una línea de fabricación. El sistema puede aprender a distinguir "pieza correcta" de "pieza defectuosa" con una sola imagen de referencia por categoría, sin parar la producción para etiquetar datasets.

**Escenarios:**
- Control de calidad visual en piezas mecánicas
- Verificación de etiquetado en packaging
- Detección de soldaduras defectuosas
- Clasificación de componentes electrónicos en SMT

---

### Análisis de señales biomédicas escaneadas

Historiales médicos en papel, electrocardiogramas impresos, espectros de laboratorio, curvas de espirometría. El pipeline convierte esas gráficas en datos numéricos procesables y sus expresiones analíticas en parámetros clínicamente interpretables.

```python
puntos = extraer_puntos("ecg_papel.png", x_min=0, x_max=10,
                         y_min=-2, y_max=2)
resultado = analizar(puntos)
# Cada tramo puede corresponder a una fase del ciclo cardíaco
```

---

### Robótica y sistemas embebidos sin GPU

Robots colaborativos, drones de inspección, cámaras de seguridad edge, sistemas de guiado en vehículos autónomos de baja potencia. La ausencia de dependencia de GPU y la latencia por debajo de 100ms en CPU estándar lo hacen viable en hardware restringido.

La biblioteca de clases se almacena en un fichero `.npz` de pocos KB, lo que permite despliegues en dispositivos con memoria muy limitada.

---

### Educación y accesibilidad

Aplicaciones de ayuda a la lectura para personas con dislexia o discapacidad visual, donde el sistema identifica letras o símbolos en tiempo real con feedback inmediato. La posibilidad de registrar variantes personalizadas de escritura permite adaptación individual sin entrenamiento generalizado.

---

### Archivo y catalogación automatizada

Digitalización de colecciones históricas: herbarios, archivos notariales, colecciones filatélicas, piezas arqueológicas. El sistema puede catalogar por similitud de forma usando solo un ejemplar representativo por categoría, sin etiquetar miles de imágenes manualmente.

---

### Calibración de modelos de visión existentes

Cuando ya existe un clasificador entrenado pero sus scores no están bien calibrados, el módulo `score_calibrator` ajusta una expresión analítica sobre la curva empírica score→precisión y la convierte en una capa Keras que corrige los scores sin reentrenar el modelo.

```python
from score_calibrator import CalibradorScores

cal = CalibradorScores(modelo_existente, umbral_r2=0.95)
cal.ajustar(X_val, y_val)
# → f(score) = 1/(1 + e^(-8.3·(s-0.61))) + 0.02
modelo_calibrado = cal.envolver_modelo()
umbral = cal.umbral_optimo(target_precision=0.90)
```

---

## Cuándo usar este sistema y cuándo no

**Usar cuando:**
- No hay datos de entrenamiento o son muy escasos (< 50 ejemplos por clase)
- El hardware no tiene GPU o tiene memoria limitada
- Las clases cambian frecuentemente y reentrenar es inviable
- La interpretabilidad y auditabilidad son un requisito
- El dominio es muy específico y no existen modelos preentrenados relevantes
- Las imágenes tienen fondo simple (documentos, pizarras, superficies uniformes)

**No usar cuando:**
- Hay miles de ejemplos etiquetados disponibles y GPU → usar EfficientDet o YOLO
- Las escenas son naturales complejas con fondos variables y oclusiones severas
- Se necesita detectar objetos muy pequeños o muy similares entre sí en detalle fino

---

## Hoja de ruta

| Estado | Componente |
|---|---|
| ✅ | Extracción de puntos de gráficas (con OCR y resistencia a grid) |
| ✅ | Motor de ajuste analítico con divide y vencerás |
| ✅ | Codificación de formas con EFD + geometría + momentos Hu |
| ✅ | Detección zero-shot + clasificación por similitud coseno |
| ✅ | Calibración analítica de scores con capa Keras |
| 🔲 | Clasificador de familia de funciones con Random Forest |
| 🔲 | Soporte para formas con múltiples trazos desconectados |
| 🔲 | Interfaz de etiquetado para refinamiento con pocos datos |
| 🔲 | Benchmarks formales contra HOG+SVM y MobileNetV3 |
| 🔲 | Exportación a ONNX para despliegue en edge devices |
