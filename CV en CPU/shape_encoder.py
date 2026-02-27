"""
====================================================
  SHAPE ENCODER  —  Etapa 1 del pipeline
====================================================
  Convierte una forma (letra, dibujo, símbolo) en
  un vector de parámetros analíticos extraídos de
  su contorno paramétrico (x(t), y(t)).

  Ese vector puede usarse como:
    - Feature para un clasificador
    - Descriptor para comparar formas
    - Representación compacta e interpretable

  Dependencias:
    pip install opencv-python numpy scipy matplotlib

  Uso:
    from shape_encoder import codificar_forma
    features = codificar_forma("letra_a.png")
    # → dict con parámetros analíticos + descriptores
====================================================
"""

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# Motor analítico propio
from analytical_engine import analizar, R2_ACEPTABLE


# ════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ════════════════════════════════════════════════════

N_MUESTRAS_T   = 512    # resolución de parametrización
N_FOURIER      = 16     # armónicos para descriptor EFD
MIN_AREA_CONTORNO = 100 # área mínima de contorno válido (px²)
TAM_NORMALIZADO   = 128 # tamaño de imagen normalizada


# ════════════════════════════════════════════════════
#  PASO 1: PREPROCESADO DE LA IMAGEN DE FORMA
# ════════════════════════════════════════════════════

def preprocesar_forma(ruta_o_array, pad: int = 10) -> np.ndarray:
    """
    Carga y normaliza la imagen de la forma:
      - Escala de grises
      - Umbralización Otsu
      - Recorte al bounding box + padding
      - Redimensión a TAM_NORMALIZADO × TAM_NORMALIZADO

    Acepta ruta (str) o array numpy (BGR o gris).
    Retorna imagen binaria normalizada.
    """
    if isinstance(ruta_o_array, str):
        img = cv2.imread(ruta_o_array)
        if img is None:
            raise FileNotFoundError(f"No encontrado: {ruta_o_array}")
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif ruta_o_array.ndim == 3:
        gris = cv2.cvtColor(ruta_o_array, cv2.COLOR_BGR2GRAY)
    else:
        gris = ruta_o_array.copy()

    # Invertir si fondo oscuro (la forma debe ser blanca en negro)
    if np.mean(gris) < 127:
        gris = cv2.bitwise_not(gris)

    # Umbralización Otsu
    _, binaria = cv2.threshold(gris, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Recortar al bounding box
    coords = cv2.findNonZero(binaria)
    if coords is None:
        return np.zeros((TAM_NORMALIZADO, TAM_NORMALIZADO), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    recortada  = binaria[max(0, y-pad): y+h+pad,
                         max(0, x-pad): x+w+pad]

    # Normalizar a tamaño fijo
    normalizada = cv2.resize(recortada,
                             (TAM_NORMALIZADO, TAM_NORMALIZADO),
                             interpolation=cv2.INTER_AREA)
    return normalizada


# ════════════════════════════════════════════════════
#  PASO 2: EXTRACCIÓN DEL CONTORNO PRINCIPAL
# ════════════════════════════════════════════════════

def extraer_contorno_principal(binaria: np.ndarray) -> Optional[np.ndarray]:
    """
    Extrae el contorno más largo (el que define la forma principal)
    como array de puntos Nx2 ordenados.

    Retorna array shape (N, 2) con columnas [x, y], o None.
    """
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return None

    # Filtrar por área mínima y tomar el más grande
    validos = [c for c in contornos
               if cv2.contourArea(c) >= MIN_AREA_CONTORNO]
    if not validos:
        validos = contornos  # fallback: tomar todos

    contorno = max(validos, key=cv2.contourArea)
    pts = contorno.reshape(-1, 2).astype(float)  # (N, 2)

    # Cerrar el contorno (último punto = primero)
    pts = np.vstack([pts, pts[0]])

    return pts


# ════════════════════════════════════════════════════
#  PASO 3: PARAMETRIZACIÓN (x(t), y(t))
# ════════════════════════════════════════════════════

def parametrizar_contorno(pts: np.ndarray,
                          n_muestras: int = N_MUESTRAS_T
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte la secuencia de puntos del contorno en
    curvas paramétricas continuas x(t) e y(t) usando
    la longitud de arco como parámetro t ∈ [0, 1].

    Retorna (t, x_t, y_t) — arrays de longitud n_muestras.
    """
    # Longitud de arco acumulada como parámetro natural
    deltas   = np.diff(pts, axis=0)
    distancias = np.sqrt((deltas ** 2).sum(axis=1))
    longitud  = np.concatenate([[0], np.cumsum(distancias)])
    longitud /= longitud[-1]  # normalizar a [0, 1]

    # Eliminar duplicados en t (puede ocurrir con contornos densos)
    _, idx_unicos = np.unique(longitud, return_index=True)
    longitud = longitud[idx_unicos]
    pts      = pts[idx_unicos]

    # Spline cúbica periódica para x(t) e y(t)
    cs_x = CubicSpline(longitud, pts[:, 0], bc_type="periodic"
                       if pts[0][0] == pts[-1][0] else "not-a-knot")
    cs_y = CubicSpline(longitud, pts[:, 1], bc_type="periodic"
                       if pts[0][1] == pts[-1][1] else "not-a-knot")

    t    = np.linspace(0, 1, n_muestras, endpoint=False)
    x_t  = cs_x(t)
    y_t  = cs_y(t)

    return t, x_t, y_t


# ════════════════════════════════════════════════════
#  PASO 4: REPRESENTACIÓN ANALÍTICA DE x(t) e y(t)
# ════════════════════════════════════════════════════

def representacion_analitica_parametrica(
    t   : np.ndarray,
    x_t : np.ndarray,
    y_t : np.ndarray,
    umbral_r2: float = R2_ACEPTABLE,
) -> dict:
    """
    Aplica analytical_engine sobre x(t) e y(t)
    por separado para obtener la representación analítica
    de cada componente paramétrica.

    Retorna dict con:
        x_tramos : list[AjusteParcial] para x(t)
        y_tramos : list[AjusteParcial] para y(t)
        r2_x     : R² global de x(t)
        r2_y     : R² global de y(t)
        r2_medio : promedio de ambos
    """
    puntos_x = [(float(ti), float(xi)) for ti, xi in zip(t, x_t)]
    puntos_y = [(float(ti), float(yi)) for ti, yi in zip(t, y_t)]

    resultado_x = analizar(puntos_x, umbral_r2=umbral_r2, verboso=False)
    resultado_y = analizar(puntos_y, umbral_r2=umbral_r2, verboso=False)

    return {
        "x_tramos": resultado_x.tramos,
        "y_tramos": resultado_y.tramos,
        "r2_x"    : resultado_x.r2_global(),
        "r2_y"    : resultado_y.r2_global(),
        "r2_medio": (resultado_x.r2_global() + resultado_y.r2_global()) / 2,
    }


# ════════════════════════════════════════════════════
#  PASO 5: DESCRIPTORES EFD (Elliptic Fourier)
#          como features complementarios
# ════════════════════════════════════════════════════

def descriptores_efd(pts: np.ndarray,
                     n_armonicos: int = N_FOURIER) -> np.ndarray:
    """
    Calcula los Descriptores de Fourier Elípticos (EFD).

    Son invariantes a rotación, escala y traslación
    (tras normalización), y capturan la forma global
    de manera muy compacta.

    Retorna array 1D de longitud 4 * n_armonicos
    (coeficientes an, bn, cn, dn por armónico).
    """
    dxy      = np.diff(pts, axis=0)
    dt       = np.sqrt((dxy ** 2).sum(axis=1))
    dt       = np.where(dt == 0, 1e-10, dt)
    T        = dt.sum()
    t_acum   = np.concatenate([[0], np.cumsum(dt)])

    coefs = np.zeros((n_armonicos, 4))

    for n in range(1, n_armonicos + 1):
        factor = T / (2 * n**2 * np.pi**2)
        phi_n  = 2 * n * np.pi / T

        cos_diff = np.cos(phi_n * t_acum[1:]) - np.cos(phi_n * t_acum[:-1])
        sin_diff = np.sin(phi_n * t_acum[1:]) - np.sin(phi_n * t_acum[:-1])

        an = factor * np.sum(dxy[:, 0] / dt * cos_diff)
        bn = -factor * np.sum(dxy[:, 0] / dt * sin_diff)
        cn = factor * np.sum(dxy[:, 1] / dt * cos_diff)
        dn = -factor * np.sum(dxy[:, 1] / dt * sin_diff)

        coefs[n - 1] = [an, bn, cn, dn]

    # Normalizar respecto al primer armónico (invariancia a escala/rotación)
    a1, b1, c1, d1 = coefs[0]
    escala = np.sqrt(a1**2 + c1**2) + 1e-10
    angulo = np.arctan2(c1, a1)
    coefs_norm = coefs.copy()
    for n in range(n_armonicos):
        phi = (n + 1) * angulo
        rot = np.array([[np.cos(phi), np.sin(phi)],
                        [-np.sin(phi), np.cos(phi)]])
        ab = rot @ np.array([coefs[n, 0], coefs[n, 1]])
        cd = rot @ np.array([coefs[n, 2], coefs[n, 3]])
        coefs_norm[n] = [ab[0] / escala, ab[1] / escala,
                         cd[0] / escala, cd[1] / escala]

    return coefs_norm.flatten()


# ════════════════════════════════════════════════════
#  PASO 6: MÉTRICAS GEOMÉTRICAS DE LA FORMA
# ════════════════════════════════════════════════════

def metricas_geometricas(binaria: np.ndarray,
                         contorno: np.ndarray) -> dict:
    """
    Extrae métricas geométricas clásicas que complementan
    la representación analítica:

      - compacidad       : 4π·área / perímetro²
      - excentricidad    : de la elipse equivalente
      - solidez          : área / área convex hull
      - extension        : área / área bounding box
      - n_agujeros       : número de agujeros internos
      - momentos_hu      : 7 momentos de Hu (invariantes)
      - circularidad     : qué tan circular es la forma
    """
    area      = cv2.contourArea(contorno[:-1].reshape(-1, 1, 2)
                                .astype(np.int32))
    perimetro = cv2.arcLength(contorno[:-1].reshape(-1, 1, 2)
                              .astype(np.int32), closed=True)

    # Compacidad
    compacidad  = (4 * np.pi * area) / (perimetro**2 + 1e-10)

    # Convex hull y solidez
    hull         = cv2.convexHull(contorno[:-1].reshape(-1, 1, 2)
                                  .astype(np.int32))
    area_hull    = cv2.contourArea(hull) + 1e-10
    solidez      = area / area_hull

    # Bounding box y extensión
    x, y, w, h   = cv2.boundingRect(contorno[:-1].reshape(-1, 1, 2)
                                    .astype(np.int32))
    extension    = area / ((w * h) + 1e-10)
    aspect_ratio = w / (h + 1e-10)

    # Elipse equivalente y excentricidad
    excentricidad = 0.0
    if len(contorno) >= 5:
        try:
            (_, _), (ma, mi), _ = cv2.fitEllipse(
                contorno[:-1].reshape(-1, 1, 2).astype(np.int32))
            excentricidad = np.sqrt(1 - (min(ma,mi) / (max(ma,mi)+1e-10))**2)
        except Exception:
            pass

    # Agujeros internos
    contornos_todos, jerarquia = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    n_agujeros = 0
    if jerarquia is not None:
        n_agujeros = int(np.sum(jerarquia[0, :, 3] >= 0))

    # Momentos de Hu
    M = cv2.moments(contorno[:-1].reshape(-1, 1, 2).astype(np.int32))
    hu = cv2.HuMoments(M).flatten()
    # Transformación logarítmica para mejor escala
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return {
        "compacidad"  : float(compacidad),
        "excentricidad": float(excentricidad),
        "solidez"     : float(solidez),
        "extension"   : float(extension),
        "aspect_ratio": float(aspect_ratio),
        "n_agujeros"  : int(n_agujeros),
        "momentos_hu" : hu_log.tolist(),
        "area_norm"   : float(area / (TAM_NORMALIZADO**2)),
    }


# ════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ════════════════════════════════════════════════════

def codificar_forma(
    entrada          ,           # ruta str o array numpy
    umbral_r2        : float = 0.93,
    incluir_efd      : bool  = True,
    incluir_analitico: bool  = True,
    incluir_geometria: bool  = True,
    guardar_debug    : bool  = False,
    ruta_debug       : str   = "debug_shape.png",
) -> dict:
    """
    Pipeline completo: imagen de forma → vector de features.

    Retorna dict con:
        "vector"      : np.ndarray 1D listo para clasificador
        "efd"         : descriptores de Fourier elípticos
        "geometria"   : métricas geométricas
        "analitico"   : representación analítica paramétrica
        "contorno"    : puntos del contorno
        "t", "x_t", "y_t": curvas paramétricas muestreadas
        "r2_medio"    : calidad del ajuste analítico
    """
    SEP = "─" * 50
    print(f"\n{SEP}\n  SHAPE ENCODER\n{SEP}")

    # 1. Preprocesar
    binaria = preprocesar_forma(entrada)
    print(f"[1/5] Imagen normalizada: {binaria.shape}")

    # 2. Extraer contorno
    contorno = extraer_contorno_principal(binaria)
    if contorno is None or len(contorno) < 10:
        print("  ⚠ No se encontró contorno válido")
        return {"vector": np.zeros(4 * N_FOURIER + 15), "error": True}
    print(f"[2/5] Contorno: {len(contorno)} puntos")

    # 3. Parametrizar
    t, x_t, y_t = parametrizar_contorno(contorno)
    print(f"[3/5] Parametrización: {len(t)} muestras en t∈[0,1]")

    resultado = {"contorno": contorno, "t": t, "x_t": x_t, "y_t": y_t}

    # 4. Descriptores EFD
    efd = np.array([])
    if incluir_efd:
        efd = descriptores_efd(contorno, N_FOURIER)
        resultado["efd"] = efd
        print(f"[4/5] EFD: {len(efd)} coeficientes ({N_FOURIER} armónicos)")

    # 5a. Representación analítica paramétrica
    analitico = {}
    if incluir_analitico:
        print(f"[5/5] Ajuste analítico de x(t) e y(t)...")
        analitico  = representacion_analitica_parametrica(
            t, x_t, y_t, umbral_r2=umbral_r2)
        resultado["analitico"] = analitico
        print(f"  R² x(t)={analitico['r2_x']:.4f}  "
              f"y(t)={analitico['r2_y']:.4f}  "
              f"medio={analitico['r2_medio']:.4f}")

    # 5b. Métricas geométricas
    geometria = {}
    if incluir_geometria:
        geometria = metricas_geometricas(binaria, contorno)
        resultado["geometria"] = geometria

    # Construir vector de features unificado
    partes = []
    if incluir_efd and len(efd):
        partes.append(efd)
    if incluir_geometria and geometria:
        geo_vec = np.array([
            geometria["compacidad"],
            geometria["excentricidad"],
            geometria["solidez"],
            geometria["extension"],
            geometria["aspect_ratio"],
            float(geometria["n_agujeros"]),
            geometria["area_norm"],
        ] + geometria["momentos_hu"])
        partes.append(geo_vec)
    if incluir_analitico and analitico:
        # Parámetros analíticos: R² + n_tramos de cada componente
        partes.append(np.array([
            analitico["r2_x"],
            analitico["r2_y"],
            float(len(analitico["x_tramos"])),
            float(len(analitico["y_tramos"])),
        ]))

    resultado["vector"]   = np.concatenate(partes) if partes else np.array([])
    resultado["r2_medio"] = analitico.get("r2_medio", 0.0)

    if guardar_debug:
        _guardar_debug_forma(binaria, contorno, t, x_t, y_t,
                             analitico, ruta_debug)

    print(f"\n  ✓ Vector de {len(resultado['vector'])} features\n{SEP}")
    return resultado


# ════════════════════════════════════════════════════
#  DIAGNÓSTICO VISUAL
# ════════════════════════════════════════════════════

def _guardar_debug_forma(binaria, contorno, t, x_t, y_t,
                          analitico, ruta):
    """4 paneles: forma, contorno, x(t), y(t) con ajuste."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: forma binaria
    axes[0].imshow(binaria, cmap="gray")
    axes[0].set_title("1. Forma normalizada")
    axes[0].axis("off")

    # Panel 2: contorno sobre la forma
    axes[1].imshow(binaria, cmap="gray", alpha=0.5)
    axes[1].plot(contorno[:, 0], contorno[:, 1], "r-", lw=1.5)
    axes[1].set_title(f"2. Contorno ({len(contorno)} pts)")
    axes[1].axis("off")

    # Panel 3: x(t) con ajuste analítico
    axes[2].plot(t, x_t, "gray", lw=1, alpha=0.7, label="x(t) real")
    if analitico and analitico.get("x_tramos"):
        for tr in analitico["x_tramos"]:
            t_d = np.linspace(tr.x_inicio, tr.x_fin, 200)
            y_d = tr.funcion(t_d, *list(tr.parametros.values()))
            axes[2].plot(t_d, y_d, "b-", lw=2)
    axes[2].set_title(f"3. x(t)  R²={analitico.get('r2_x', 0):.3f}")
    axes[2].set_xlabel("t"); axes[2].grid(True, alpha=0.3)

    # Panel 4: y(t) con ajuste analítico
    axes[3].plot(t, y_t, "gray", lw=1, alpha=0.7, label="y(t) real")
    if analitico and analitico.get("y_tramos"):
        for tr in analitico["y_tramos"]:
            t_d = np.linspace(tr.x_inicio, tr.x_fin, 200)
            y_d = tr.funcion(t_d, *list(tr.parametros.values()))
            axes[3].plot(t_d, y_d, "r-", lw=2)
    axes[3].set_title(f"4. y(t)  R²={analitico.get('r2_y', 0):.3f}")
    axes[3].set_xlabel("t"); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Debug] {ruta}")


# ════════════════════════════════════════════════════
#  UTILIDAD: BATCH sobre un dataset
# ════════════════════════════════════════════════════

def codificar_dataset(rutas: list[str],
                      etiquetas: Optional[list] = None,
                      **kwargs) -> tuple[np.ndarray, Optional[list]]:
    """
    Codifica una lista de imágenes de formas en una
    matriz de features lista para Keras/sklearn.

    Retorna (X, y) donde:
        X : np.ndarray shape (N, n_features)
        y : lista de etiquetas o None
    """
    vectores = []
    for i, ruta in enumerate(rutas):
        print(f"\n  [{i+1}/{len(rutas)}] {ruta}")
        try:
            resultado = codificar_forma(ruta, **kwargs)
            vectores.append(resultado["vector"])
        except Exception as e:
            print(f"  ⚠ Error en {ruta}: {e}")
            vectores.append(np.zeros_like(vectores[0])
                            if vectores else np.zeros(80))

    X = np.vstack(vectores)
    return X, etiquetas


# ════════════════════════════════════════════════════
#  EJEMPLO DE USO
# ════════════════════════════════════════════════════
if __name__ == "__main__":

    # Ejemplo A — una sola imagen
    resultado = codificar_forma(
        "letra_a.png",
        umbral_r2    = 0.93,
        incluir_efd  = True,
        guardar_debug= True,
    )
    print(f"\nVector de features: shape={resultado['vector'].shape}")
    print(f"R² ajuste analítico: {resultado['r2_medio']:.4f}")

    # Ejemplo B — dataset completo
    # import glob
    # rutas     = glob.glob("dataset/letras/*.png")
    # etiquetas = [r.split("/")[-1][0] for r in rutas]  # primera letra del nombre
    # X, y = codificar_dataset(rutas, etiquetas)
    # np.save("features_formas.npy", X)
