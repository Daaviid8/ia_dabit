"""
====================================================
  EXTRACTOR ROBUSTO DE PUNTOS DE GRÁFICAS  v2
====================================================
  Maneja:
    ✓ Fondo blanco sin cuadrícula
    ✓ Fondo con cuadrícula (grid)
    ✓ Curvas negras / grises
    ✓ Curvas de color sólido (rojo, azul, verde...)
    ✓ Ejes con marcas y números (OCR automático)
    ✓ Ejes sin marcas (calibración manual)
    ✓ Curvas que se cortan con los ejes (reconstrucción
      direccional en los cruces)

  Dependencias:
    pip install opencv-python numpy scipy matplotlib
    pip install pytesseract pillow          ← para OCR
    + instalar Tesseract en el sistema:
        Windows : https://github.com/UB-Mannheim/tesseract/wiki
        Linux   : sudo apt install tesseract-ocr
        macOS   : brew install tesseract

  Uso mínimo:
    puntos = extraer_puntos("grafica.png")

  Con calibración manual (si no hay números en los ejes):
    puntos = extraer_puntos("grafica.png",
                            x_min=-5, x_max=5,
                            y_min=-3, y_max=3)
====================================================
"""

import cv2
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional
import warnings
import os
warnings.filterwarnings("ignore")

# OCR opcional
try:
    import pytesseract
    from PIL import Image as PILImage
    OCR_DISPONIBLE = True
except ImportError:
    OCR_DISPONIBLE = False


# ════════════════════════════════════════════════════
#  CONSTANTES DE CONFIGURACIÓN
# ════════════════════════════════════════════════════

UMBRAL_OSCURO  = 80    # gris máximo para considerar píxel "oscuro"
MIN_AREA_CURVA = 200   # área mínima de componente conectada (px²)
TOL_EJE        = 4     # tolerancia en px para enmascarar ejes
MAX_HUECO      = 15    # hueco horizontal máximo a interpolar (px)
MIN_PUNTOS     = 5     # mínimo de puntos para tramo válido


# ════════════════════════════════════════════════════
#  PASO 1: CARGA Y NORMALIZACIÓN
# ════════════════════════════════════════════════════

def cargar_y_normalizar(ruta: str) -> tuple:
    """
    Carga la imagen y la redimensiona si es muy grande.
    Retorna (img_bgr, img_gris).
    """
    img = cv2.imread(ruta)
    if img is None:
        raise FileNotFoundError(f"No se encontró: {ruta}")

    MAX_DIM = 1400
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        factor = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * factor), int(h * factor)),
                         interpolation=cv2.INTER_AREA)

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gris


# ════════════════════════════════════════════════════
#  PASO 2: DETECCIÓN DE EJES (en cascada)
# ════════════════════════════════════════════════════

def detectar_ejes(img_bgr: np.ndarray, gris: np.ndarray) -> dict:
    """
    Detecta los ejes X (horizontal) e Y (vertical).

    Cascada:
      1. Hough sobre bordes Canny → líneas largas
      2. Proyección de densidad → filas/columnas densas
      3. Fallback por margen fijo

    Retorna dict: eje_x, eje_y, origen, alto, ancho
    """
    h, w = gris.shape
    bordes = cv2.Canny(gris, 40, 120)
    min_len = min(w, h) // 5
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, 80,
                             minLineLength=min_len, maxLineGap=8)

    horiz, vert = [], []
    if lineas is not None:
        for l in lineas:
            x1, y1, x2, y2 = l[0]
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang < 6:
                horiz.append(y1)
            elif ang > 84:
                vert.append(x1)

    # Mediana de líneas detectadas
    eje_x = int(np.median(horiz)) if horiz else None
    eje_y = int(np.median(vert))  if vert  else None

    # Fallback: proyección de densidad
    binaria = cv2.adaptiveThreshold(gris, 255,
                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                 cv2.THRESH_BINARY_INV, 15, 3)

    if eje_x is None:
        proj_h = np.sum(binaria, axis=1).astype(float)
        zona_h = proj_h[h // 3:]
        idx = int(np.argmax(zona_h)) + h // 3
        eje_x = idx if proj_h[idx] > w * 0.25 * 255 else int(h * 0.82)

    if eje_y is None:
        proj_v = np.sum(binaria, axis=0).astype(float)
        zona_v = proj_v[:w // 2]
        idx = int(np.argmax(zona_v))
        eje_y = idx if proj_v[idx] > h * 0.25 * 255 else int(w * 0.10)

    return {
        "eje_x" : int(eje_x),
        "eje_y" : int(eje_y),
        "origen": (int(eje_y), int(eje_x)),
        "alto"  : h,
        "ancho" : w,
    }


# ════════════════════════════════════════════════════
#  PASO 3: CALIBRACIÓN (OCR o manual)
# ════════════════════════════════════════════════════

def calibrar_ejes(img_bgr, ejes, x_min, x_max, y_min, y_max,
                  intentar_ocr=True) -> dict:
    """
    Determina los rangos reales de los ejes.
    Prioridad: valores manuales > OCR > defaults [0,1].
    """
    if all(v is not None for v in [x_min, x_max, y_min, y_max]):
        print("  [Calibración] Valores manuales")
        return dict(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    if intentar_ocr and OCR_DISPONIBLE:
        resultado = _ocr_calibracion(img_bgr, ejes)
        if resultado:
            print(f"  [Calibración] OCR → {resultado}")
            resultado["x_min"] = x_min if x_min is not None else resultado["x_min"]
            resultado["x_max"] = x_max if x_max is not None else resultado["x_max"]
            resultado["y_min"] = y_min if y_min is not None else resultado["y_min"]
            resultado["y_max"] = y_max if y_max is not None else resultado["y_max"]
            return resultado

    if intentar_ocr and not OCR_DISPONIBLE:
        print("  [Calibración] pytesseract no instalado")

    print("  [Calibración] Usando [0,1] × [0,1] por defecto")
    return dict(
        x_min=x_min if x_min is not None else 0.0,
        x_max=x_max if x_max is not None else 1.0,
        y_min=y_min if y_min is not None else 0.0,
        y_max=y_max if y_max is not None else 1.0,
    )


def _ocr_calibracion(img_bgr, ejes) -> Optional[dict]:
    """Usa Tesseract para leer los números de los márgenes."""
    import re
    h, w   = img_bgr.shape[:2]
    ex, ey = ejes["eje_x"], ejes["eje_y"]
    margen = 55
    cfg    = "--psm 6 -c tessedit_char_whitelist=0123456789.,-"

    def leer_nums(region):
        gris  = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        grande = cv2.resize(gris, None, fx=3, fy=3,
                            interpolation=cv2.INTER_CUBIC)
        _, bin_ = cv2.threshold(grande, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        texto = pytesseract.image_to_string(
            PILImage.fromarray(bin_), config=cfg)
        return [float(t) for t in re.findall(r"-?\d+\.?\d*", texto)
                if t]

    try:
        nums_x = leer_nums(img_bgr[ex:min(ex+margen,h), ey:w])
        nums_y = leer_nums(img_bgr[0:ex, 0:max(ey-2, margen)])
    except Exception as e:
        print(f"  [OCR] Error: {e}")
        return None

    if len(nums_x) >= 2 and len(nums_y) >= 2:
        return dict(x_min=min(nums_x), x_max=max(nums_x),
                    y_min=min(nums_y), y_max=max(nums_y))
    return None


# ════════════════════════════════════════════════════
#  PASO 4: ELIMINACIÓN DE CUADRÍCULA
# ════════════════════════════════════════════════════

def eliminar_cuadricula(gris: np.ndarray, ejes: dict) -> np.ndarray:
    """
    Detecta y elimina líneas de cuadrícula mediante:
      1. Morfología (erosión lineal) para aislar líneas rectas largas
      2. Filtrado por intensidad: las líneas del grid son más claras
         que la curva principal (la curva tiene grises < 80)
      3. Inpainting sobre las líneas detectadas

    Si no hay grid, la imagen sale intacta.
    """
    h, w = gris.shape

    binaria = cv2.adaptiveThreshold(gris, 255,
                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                 cv2.THRESH_BINARY_INV, 15, 3)

    def detectar_lineas_grid(kernel_shape):
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
        lineas = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kern)
        # Filtrar: solo las que tienen intensidad media > UMBRAL_OSCURO
        # (la curva real es muy oscura, el grid es gris claro)
        resultado = np.zeros_like(lineas)
        n_comp, etiq, stats, _ = cv2.connectedComponentsWithStats(
            lineas, connectivity=8)
        for i in range(1, n_comp):
            mask_c = (etiq == i).astype(np.uint8) * 255
            intens  = cv2.mean(gris, mask=mask_c)[0]
            if UMBRAL_OSCURO < intens < 220:   # gris medio = grid
                resultado = cv2.bitwise_or(resultado, mask_c)
        return resultado

    lines_h = detectar_lineas_grid((w // 5, 1))
    lines_v = detectar_lineas_grid((1, h // 5))

    mask_grid = cv2.bitwise_or(lines_h, lines_v)
    kern_dil  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask_grid = cv2.dilate(mask_grid, kern_dil, iterations=1)

    n_px = int(np.sum(mask_grid > 0))
    if n_px > 0:
        resultado = cv2.inpaint(gris, mask_grid, 3, cv2.INPAINT_TELEA)
        print(f"  [Grid] Eliminadas ~{n_px} px de cuadrícula")
    else:
        resultado = gris.copy()
        print("  [Grid] Sin cuadrícula detectada")

    return resultado


# ════════════════════════════════════════════════════
#  PASO 5: AISLAMIENTO DE LA CURVA
# ════════════════════════════════════════════════════

def detectar_modo(img_bgr: np.ndarray, ejes: dict) -> str:
    """Decide si la curva es 'oscura' o 'color' analizando la saturación."""
    ex, ey = ejes["eje_x"], ejes["eje_y"]
    region = img_bgr[0:ex, ey:]
    if region.size == 0:
        return "oscura"
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pct = np.sum(hsv[:, :, 1] > 80) / hsv[:, :, 1].size
    modo = "color" if pct > 0.005 else "oscura"
    print(f"  [Modo] {'Color' if modo=='color' else 'Oscura'} "
          f"({pct*100:.2f}% px saturados)")
    return modo


def aislar_curva(img_bgr, gris_sin_grid, ejes, modo) -> np.ndarray:
    """
    Genera la máscara binaria de la curva según el modo.

    Modo 'oscura': umbralización adaptativa de la imagen en gris.
    Modo 'color' : detección por tono dominante en HSV + fallback oscuro.
    """
    if modo == "color":
        mascara = _mask_color(img_bgr, ejes)
    else:
        mascara = _mask_oscura(gris_sin_grid)

    return _limpiar_mascara(mascara, ejes, gris_original=gris_sin_grid)


def _mask_oscura(gris: np.ndarray) -> np.ndarray:
    """Máscara para curva negra/gris usando umbral adaptativo."""
    return cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5)


def _mask_color(img_bgr: np.ndarray, ejes: dict) -> np.ndarray:
    """
    Máscara para curva de color.
    Encuentra el tono dominante con saturación alta en la región
    del gráfico y crea una máscara con ±18° de tolerancia en HSV.
    """
    ex, ey = ejes["eje_x"], ejes["eje_y"]
    hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    region = hsv[0:ex, ey:]
    if region.size == 0:
        return _mask_oscura(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    sat   = region[:, :, 1]
    mask_sat = (sat > 70).astype(np.uint8) * 255
    hist_h   = cv2.calcHist([region], [0], mask_sat, [180], [0, 180])
    tono     = int(np.argmax(hist_h))
    tol      = 18
    print(f"  [Color] Tono dominante H={tono}°")

    return cv2.inRange(
        hsv,
        np.array([max(0, tono - tol),  60,  40]),
        np.array([min(179, tono + tol), 255, 255]),
    )


def _construir_mascara_ejes(gris: np.ndarray, ejes: dict) -> np.ndarray:
    """
    Detecta morfológicamente los píxeles que pertenecen
    SOLO a las líneas de los ejes, usando erosión lineal
    de longitud máxima.

    Esto permite distinguir "píxel de eje" de "píxel de
    curva cruzando el eje", que es imposible con una franja fija.
    """
    h, w = gris.shape
    ex, ey = ejes["eje_x"], ejes["eje_y"]

    _, bin_ = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY_INV)

    # Eje X: línea horizontal larga centrada en eje_x
    kern_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    eje_x_mask = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, kern_h)
    # Limitar a la banda vertical del eje X real
    banda_x = np.zeros_like(eje_x_mask)
    banda_x[max(0, ex - TOL_EJE): ex + TOL_EJE + 1, :] = \
        eje_x_mask[max(0, ex - TOL_EJE): ex + TOL_EJE + 1, :]

    # Eje Y: línea vertical larga centrada en eje_y
    kern_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 3))
    eje_y_mask = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, kern_v)
    banda_y = np.zeros_like(eje_y_mask)
    banda_y[:, max(0, ey - TOL_EJE): ey + TOL_EJE + 1] = \
        eje_y_mask[:, max(0, ey - TOL_EJE): ey + TOL_EJE + 1]

    return cv2.bitwise_or(banda_x, banda_y)


def _limpiar_mascara(mascara: np.ndarray, ejes: dict,
                     gris_original: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Post-procesa la máscara binaria de la curva.

    Estrategia resistente a cruces con ejes:
      1. Construir máscara morfológica de los ejes
      2. SOLO restar los píxeles que son "eje puro":
         - Si una columna/fila tiene píxeles del eje en toda
           su longitud → eje puro → se elimina
         - Si solo tiene píxeles en la zona del cruce → puede
           ser curva → se conserva
      3. Recortar a región interior
      4. Cierre morfológico + filtro de área
    """
    ex, ey = ejes["eje_x"], ejes["eje_y"]
    h, w   = mascara.shape

    # ── 1. Máscara de ejes morfológica ───────────────
    if gris_original is not None:
        mask_ejes = _construir_mascara_ejes(gris_original, ejes)
        # Restar eje a la máscara de curva
        mascara = cv2.bitwise_and(mascara,
                                   cv2.bitwise_not(mask_ejes))
    else:
        # Fallback: borrar solo 2px (mínimo indispensable)
        mascara[ex - 2: ex + 3, :] = 0
        mascara[:, ey - 2: ey + 3] = 0

    # ── 2. Recortar a región interior del plot ────────
    mascara[:, : max(0, ey - TOL_EJE)] = 0   # izq. del eje Y
    mascara[ex + TOL_EJE + 1:, :]     = 0    # bajo el eje X

    # ── 3. Cierre morfológico (micro-huecos) ──────────
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kern)

    # ── 4. Filtrar componentes pequeñas (ruido) ───────
    n_comp, etiq, stats, _ = cv2.connectedComponentsWithStats(
        mascara, connectivity=8)
    limpia = np.zeros_like(mascara)
    for i in range(1, n_comp):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA_CURVA:
            limpia[etiq == i] = 255

    return limpia


# ════════════════════════════════════════════════════
#  PASO 6: EXTRACCIÓN SUBPÍXEL Y RECONSTRUCCIÓN
#          DE CRUCES CON LOS EJES
# ════════════════════════════════════════════════════

def extraer_pixeles(mascara: np.ndarray, ejes: dict) -> list:
    """
    1. Centroide vertical por columna (precisión subpíxel)
    2. Clasifica cada hueco horizontal como:
         - "cruce_eje"  : el hueco coincide con la banda
                          de un eje → reconstruir con cúbica
         - "hueco_menor": hueco pequeño (<= MAX_HUECO) por
                          marca, tick, etc. → interpolar lineal
         - "discontinuidad": hueco grande sin eje → dejar tal cual
    3. Rellena los huecos reconstruibles con spline cúbica
       usando los N vecinos más cercanos a cada lado.

    Retorna lista de (col_float, fila_float).
    """
    from scipy.interpolate import CubicSpline

    ey = ejes["eje_y"]
    ex = ejes["eje_x"]
    h, w = mascara.shape

    # Bandas donde los ejes borran píxeles
    BANDA_X = (max(0, ex - TOL_EJE - 2), ex + TOL_EJE + 3)   # filas del eje X
    BANDA_Y = (max(0, ey - TOL_EJE - 2), ey + TOL_EJE + 3)   # cols del eje Y

    # ── Centroide por columna ─────────────────────────
    raw = {}
    for col in range(ey + TOL_EJE, w):
        filas = np.where(mascara[:, col] > 0)[0]
        if len(filas) > 0:
            raw[col] = float(np.mean(filas))

    if not raw:
        return []

    cols_validas = sorted(raw.keys())

    # ── Identificar y rellenar huecos ────────────────
    final = dict(raw)
    N_VECINOS = 6   # puntos a cada lado para ajustar la cúbica

    for i in range(len(cols_validas) - 1):
        c1 = cols_validas[i]
        c2 = cols_validas[i + 1]
        hueco = c2 - c1 - 1
        if hueco <= 0:
            continue

        # Determinar tipo de hueco
        cols_hueco = range(c1 + 1, c2)
        cruce_eje_y = any(BANDA_Y[0] <= c <= BANDA_Y[1] for c in cols_hueco)
        cruce_eje_x = (BANDA_X[0] <= raw[c1] <= BANDA_X[1] or
                       BANDA_X[0] <= raw[c2] <= BANDA_X[1])

        es_cruce = cruce_eje_y or cruce_eje_x
        es_hueco_menor = hueco <= MAX_HUECO

        if not es_cruce and not es_hueco_menor:
            continue   # discontinuidad real → no tocar

        if es_cruce:
            # Reconstrucción cúbica: tomar N vecinos a cada lado
            idx1 = i
            idx2 = i + 1
            izq = [cols_validas[max(0, idx1 - k)]
                   for k in range(N_VECINOS - 1, -1, -1)
                   if max(0, idx1 - k) >= 0]
            der = [cols_validas[min(len(cols_validas)-1, idx2 + k)]
                   for k in range(N_VECINOS)
                   if min(len(cols_validas)-1, idx2+k) < len(cols_validas)]

            pts_x = [c for c in izq + der if c in raw]
            pts_y = [raw[c] for c in pts_x]

            if len(pts_x) >= 4:
                # Eliminar duplicados y ordenar
                pares = sorted(set(zip(pts_x, pts_y)), key=lambda p: p[0])
                px = np.array([p[0] for p in pares])
                py = np.array([p[1] for p in pares])
                try:
                    cs = CubicSpline(px, py)
                    for c in cols_hueco:
                        y_interp = float(cs(c))
                        # Sanity check: no salirse demasiado del rango local
                        y_local_min = min(raw[c1], raw[c2]) - abs(raw[c2] - raw[c1]) * 2
                        y_local_max = max(raw[c1], raw[c2]) + abs(raw[c2] - raw[c1]) * 2
                        if y_local_min <= y_interp <= y_local_max:
                            final[c] = y_interp
                        else:
                            # Fallback lineal si la cúbica se dispara
                            t = (c - c1) / (c2 - c1)
                            final[c] = (1 - t) * raw[c1] + t * raw[c2]
                except Exception:
                    # Fallback lineal
                    for c in cols_hueco:
                        t = (c - c1) / (c2 - c1)
                        final[c] = (1 - t) * raw[c1] + t * raw[c2]
            else:
                # Muy pocos vecinos → lineal
                for c in cols_hueco:
                    t = (c - c1) / (c2 - c1)
                    final[c] = (1 - t) * raw[c1] + t * raw[c2]

        else:
            # Hueco menor: interpolación lineal simple
            for c in cols_hueco:
                t = (c - c1) / (c2 - c1)
                final[c] = (1 - t) * raw[c1] + t * raw[c2]

    return [(col, fila) for col, fila in sorted(final.items())]


# ════════════════════════════════════════════════════
#  PASO 7: PÍXELES → COORDENADAS REALES
# ════════════════════════════════════════════════════

def pixeles_a_coordenadas(pixeles, ejes, calibracion) -> list:
    """
    Mapea (col, fila) → (x, y) real usando el origen de los ejes
    como referencia y la calibración para escalar.
    """
    ex, ey   = ejes["eje_x"], ejes["eje_y"]
    ancho, alto = ejes["ancho"], ejes["alto"]
    xn, xx   = calibracion["x_min"], calibracion["x_max"]
    yn, yx   = calibracion["y_min"], calibracion["y_max"]

    px_x = (ancho - ey) or 1
    px_y = ex or 1
    esc_x = (xx - xn) / px_x
    esc_y = (yx - yn) / px_y

    puntos = []
    for col, fila in pixeles:
        x_r = xn + (col  - ey) * esc_x
        y_r = yn + (ex - fila) * esc_y
        puntos.append((round(float(x_r), 5), round(float(y_r), 5)))

    return sorted(puntos, key=lambda p: p[0])


# ════════════════════════════════════════════════════
#  PASO 8: SUAVIZADO ADAPTATIVO
# ════════════════════════════════════════════════════

def suavizar(puntos, fuerza=2) -> list:
    """
    Filtro Savitzky-Golay con ventana adaptada al nivel elegido.
    fuerza: 0=ninguno, 1=leve, 2=normal, 3=fuerte
    """
    ventanas = {1: 7, 2: 15, 3: 31}
    v = ventanas.get(fuerza)
    n = len(puntos)
    if not v or n < v + 2:
        return puntos

    v = min(v, n if n % 2 == 1 else n - 1)
    if v < 5:
        return puntos

    xs = np.array([p[0] for p in puntos])
    ys = np.array([p[1] for p in puntos])
    ys_s = savgol_filter(ys, window_length=v, polyorder=3)
    return [(x, round(float(y), 5)) for x, y in zip(xs, ys_s)]


# ════════════════════════════════════════════════════
#  DIAGNÓSTICO VISUAL (4 paneles)
# ════════════════════════════════════════════════════

def guardar_debug(img_bgr, gris_sin_grid, mascara, ejes, puntos,
                  ruta="debug_extraccion.png"):
    """
    Guarda una imagen con 4 paneles de diagnóstico:
      1. Original con ejes marcados
      2. Sin cuadrícula
      3. Máscara de la curva (mapa de calor)
      4. Puntos extraídos como gráfica
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ex, ey = ejes["eje_x"], ejes["eje_y"]

    # Panel 1
    orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig)
    axes[0].axhline(ex, c="red",  lw=1.5, label=f"Eje X (y={ex}px)")
    axes[0].axvline(ey, c="blue", lw=1.5, label=f"Eje Y (x={ey}px)")
    axes[0].plot(*ejes["origen"], "go", ms=8, label="Origen")
    axes[0].set_title("1. Original + ejes detectados", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].axis("off")

    # Panel 2
    axes[1].imshow(gris_sin_grid, cmap="gray")
    axes[1].set_title("2. Sin cuadrícula", fontsize=9)
    axes[1].axis("off")

    # Panel 3
    axes[2].imshow(mascara, cmap="inferno")
    n_px = int(np.sum(mascara > 0))
    axes[2].set_title(f"3. Máscara de curva\n({n_px} px activos)", fontsize=9)
    axes[2].axis("off")

    # Panel 4 — puntos extraídos, destacando zonas reconstruidas
    if puntos:
        xs = [p[0] for p in puntos]
        ys = [p[1] for p in puntos]
        axes[3].plot(xs, ys, "b-", lw=1.5, label="Curva")

        # Marcar la banda de los ejes en coordenadas reales
        cal_x_rng = (ejes["eje_y"], ejes["ancho"])
        cal_y_rng = (0, ejes["eje_x"])
        # Líneas de los ejes en espacio real (aprox x=0, y=0)
        axes[3].axvline(0, color="gray", ls="--", lw=0.8, alpha=0.6,
                        label="Eje X/Y (x=0 ó y=0)")
        axes[3].axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)

        step = max(1, len(xs) // 50)
        axes[3].scatter(xs[::step], ys[::step],
                        c="red", s=12, zorder=5, label="Muestras")
        axes[3].legend(fontsize=7)
    axes[3].set_title(f"4. Puntos extraídos (n={len(puntos)})", fontsize=9)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("x"); axes[3].set_ylabel("y")

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Debug] Imagen guardada: {ruta}")


# ════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ════════════════════════════════════════════════════

def extraer_puntos(
    ruta_imagen      : str,
    x_min            : Optional[float] = None,
    x_max            : Optional[float] = None,
    y_min            : Optional[float] = None,
    y_max            : Optional[float] = None,
    suavizado        : int  = 2,
    intentar_ocr     : bool = True,
    guardar_debug_img: bool = True,
    ruta_debug       : str  = "debug_extraccion.png",
) -> list:
    """
    Pipeline completo imagen → puntos (x, y).

    Parámetros
    ----------
    ruta_imagen      : ruta al PNG o JPG
    x_min / x_max    : rango real del eje X (None = autodetectar)
    y_min / y_max    : rango real del eje Y (None = autodetectar)
    suavizado        : 0 sin suavizado … 3 máximo
    intentar_ocr     : leer números de ejes con Tesseract
    guardar_debug_img: guardar imagen de diagnóstico
    ruta_debug       : nombre del archivo de diagnóstico

    Retorna
    -------
    Lista de (x, y) ordenada por x.
    """
    SEP = "─" * 52
    print(f"\n{SEP}\n  EXTRACTOR  ·  {os.path.basename(ruta_imagen)}\n{SEP}")

    # 1. Carga
    img, gris = cargar_y_normalizar(ruta_imagen)
    print(f"[1/7] Imagen: {img.shape[1]}×{img.shape[0]} px")

    # 2. Ejes
    ejes = detectar_ejes(img, gris)
    print(f"[2/7] Ejes → X={ejes['eje_x']}px  Y={ejes['eje_y']}px  "
          f"origen={ejes['origen']}")

    # 3. Calibración
    cal = calibrar_ejes(img, ejes, x_min, x_max, y_min, y_max, intentar_ocr)
    print(f"[3/7] Rango X:[{cal['x_min']}, {cal['x_max']}]  "
          f"Y:[{cal['y_min']}, {cal['y_max']}]")

    # 4. Eliminar grid
    gris_ng = eliminar_cuadricula(gris, ejes)
    print(f"[4/7] Cuadrícula procesada")

    # 5. Aislar curva
    modo    = detectar_modo(img, ejes)
    mascara = aislar_curva(img, gris_ng, ejes, modo)
    n_px    = int(np.sum(mascara > 0))
    print(f"[5/7] Máscara: {n_px} px activos")
    if n_px < 20:
        print("  ⚠ Muy pocos píxeles activos. Revisa la imagen y parámetros.")

    # 6. Extraer
    pixeles = extraer_pixeles(mascara, ejes)
    n_raw = sum(1 for _ in pixeles if True)  # total incluyendo reconstruidos
    print(f"[6/7] Píxeles extraídos: {n_raw} "
          f"(incluye reconstrucción en cruces de ejes)")

    # 7. Convertir + suavizar
    puntos = pixeles_a_coordenadas(pixeles, ejes, cal)
    puntos = suavizar(puntos, fuerza=suavizado)
    print(f"[7/7] Puntos finales: {len(puntos)}")

    if guardar_debug_img:
        guardar_debug(img, gris_ng, mascara, ejes, puntos, ruta_debug)

    print(f"\n  ✓ Listo\n{SEP}")
    return puntos


# ════════════════════════════════════════════════════
#  UTILIDADES
# ════════════════════════════════════════════════════

def guardar_csv(puntos, ruta="puntos.csv"):
    """Exporta la lista de puntos a CSV."""
    import csv
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        w.writerows(puntos)
    print(f"CSV guardado: {ruta}  ({len(puntos)} filas)")


def graficar_puntos(puntos, titulo="Puntos extraídos",
                    ruta="puntos_extraidos.png"):
    """Guarda una gráfica rápida de verificación."""
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, "b-", lw=1.5)
    step = max(1, len(xs) // 60)
    plt.scatter(xs[::step], ys[::step], c="red", s=18, zorder=5)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(titulo); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"Gráfica guardada: {ruta}")


# ════════════════════════════════════════════════════
#  EJEMPLO DE USO
# ════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Caso A: gráfica con grid, curva azul, con números en ejes
    puntos = extraer_puntos(
        "mi_grafica.png",
        intentar_ocr     = True,   # leer los números automáticamente
        suavizado        = 2,
        guardar_debug_img= True,
    )

    # ── Caso B: gráfica limpia, curva negra, sin marcas → manual
    # puntos = extraer_puntos(
    #     "grafica_limpia.png",
    #     x_min=-10, x_max=10,
    #     y_min=-5,  y_max=5,
    #     intentar_ocr=False,
    #     suavizado=1,
    # )

    guardar_csv(puntos, "puntos.csv")
    graficar_puntos(puntos)
