"""
====================================================
  DETECTOR + CLASIFICADOR  —  Etapa 3 del pipeline
====================================================
  Zero-shot: no necesita datos de entrenamiento.
  Clases extensibles en runtime sin reentrenar.
  Optimizado para CPU con latencia < 100ms.

  Flujo:
    OFFLINE: registrar 1 imagen de referencia por clase
    ONLINE:  detectar regiones → clasificar → calibrar

  Dependencias:
    pip install opencv-python numpy scipy scikit-learn

  Uso rápido:
    from detector_classifier import SistemaDeteccion

    sistema = SistemaDeteccion()
    sistema.registrar_clase("A", "refs/letra_a.png")
    sistema.registrar_clase("B", "refs/letra_b.png")

    resultados = sistema.detectar("documento.png")
    sistema.visualizar("documento.png", resultados)
====================================================
"""

import cv2
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass, field
from typing import Optional
import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Módulos propios
from shape_encoder import (
    codificar_forma, preprocesar_forma,
    extraer_contorno_principal, descriptores_efd,
    metricas_geometricas, N_FOURIER
)

import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════
#  CONFIGURACIÓN DE RENDIMIENTO
# ════════════════════════════════════════════════════

# Tamaño máximo de imagen antes de procesar (px lado mayor)
MAX_DIM_PROCESO    = 800

# Filtros de regiones de interés (ROI)
MIN_AREA_ROI       = 80      # px² mínimos para considerar un ROI
MAX_AREA_ROI_FRAC  = 0.85    # fracción máxima del área total
MIN_ASPECT         = 0.08    # aspect ratio mínimo (w/h)
MAX_ASPECT         = 12.0    # aspect ratio máximo

# Clasificación
UMBRAL_SIMILITUD   = 0.45    # similitud coseno mínima para asignar clase
TOP_K              = 3       # cuántas clases candidatas reportar
N_REFS_POR_CLASE   = 5       # máximo de referencias por clase (promedio)

# Rendimiento
ESCALA_RAPIDA      = 0.5     # escalar imagen para detección rápida
NMS_IOU_UMBRAL     = 0.35    # IoU para Non-Maximum Suppression


# ════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ════════════════════════════════════════════════════

@dataclass
class Deteccion:
    """Una detección individual en la imagen."""
    clase       : str
    confianza   : float          # similitud calibrada [0, 1]
    bbox        : tuple          # (x, y, w, h) en píxeles originales
    centro      : tuple          # (cx, cy)
    candidatos  : list           # [(clase, score), ...] top-K
    tiempo_ms   : float = 0.0

    def __repr__(self):
        return (f"Deteccion(clase='{self.clase}', "
                f"conf={self.confianza:.3f}, "
                f"bbox={self.bbox})")


@dataclass
class BibliotecaClase:
    """Referencia analítica de una clase."""
    nombre      : str
    vectores    : list           # lista de np.ndarray (una por imagen de ref)
    vector_medio: Optional[np.ndarray] = None
    n_ejemplos  : int = 0

    def actualizar_medio(self):
        if self.vectores:
            self.vector_medio = np.mean(self.vectores, axis=0)
            self.n_ejemplos   = len(self.vectores)


# ════════════════════════════════════════════════════
#  PASO 1: DETECCIÓN DE REGIONES (rápida, sin modelo)
# ════════════════════════════════════════════════════

def detectar_regiones(
    img_bgr    : np.ndarray,
    escala     : float = 1.0,
) -> list[tuple]:
    """
    Detecta regiones de interés (ROI) en imágenes sobre
    fondo simple (documentos, pizarras) usando contornos.

    Estrategia de 3 capas para máxima cobertura:
      1. Umbralización adaptativa → objetos oscuros sobre claro
      2. Detección de bordes Canny → objetos de cualquier color
      3. Threshold de color por canal → objetos de color

    Aplica filtros de área, aspect ratio y NMS para
    quedarse solo con los ROIs más probables.

    Retorna lista de (x, y, w, h) en coordenadas originales.
    """
    h_orig, w_orig = img_bgr.shape[:2]

    # Escalar para velocidad
    if escala != 1.0:
        w_s = int(w_orig * escala)
        h_s = int(h_orig * escala)
        img = cv2.resize(img_bgr, (w_s, h_s), interpolation=cv2.INTER_AREA)
    else:
        img = img_bgr.copy()

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gris.shape
    area_total = h * w

    mascaras = []

    # ── Capa 1: umbral adaptativo ─────────────────────
    mask1 = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 8)
    mascaras.append(mask1)

    # ── Capa 2: Canny ─────────────────────────────────
    bordes = cv2.Canny(gris, 30, 100)
    kern   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask2  = cv2.dilate(bordes, kern, iterations=2)
    mascaras.append(mask2)

    # ── Capa 3: saturación (objetos de color) ─────────
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask3 = cv2.inRange(hsv,
                         np.array([0,  60,  40]),
                         np.array([179, 255, 255]))
    mascaras.append(mask3)

    # Combinar y limpiar
    mask_total = cv2.bitwise_or(mascaras[0],
                  cv2.bitwise_or(mascaras[1], mascaras[2]))
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kern_close)

    # Extraer contornos y filtrar
    contornos, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    rois_raw = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < MIN_AREA_ROI:
            continue
        if area > area_total * MAX_AREA_ROI_FRAC:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / (bh + 1e-6)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue

        # Pequeño padding
        pad = max(3, int(min(bw, bh) * 0.08))
        x  = max(0, x - pad)
        y  = max(0, y - pad)
        bw = min(w - x, bw + 2 * pad)
        bh = min(h - y, bh + 2 * pad)

        rois_raw.append((x, y, bw, bh, float(area)))

    # NMS para eliminar solapamientos
    rois_nms = _nms(rois_raw, NMS_IOU_UMBRAL)

    # Volver a coordenadas originales
    inv = 1.0 / escala
    rois_final = [(int(x*inv), int(y*inv), int(bw*inv), int(bh*inv))
                  for x, y, bw, bh in rois_nms]

    return rois_final


def _nms(rois: list, umbral_iou: float) -> list:
    """
    Non-Maximum Suppression sobre lista de (x,y,w,h,area).
    Elimina ROIs muy solapados, conservando el de mayor área.
    """
    if not rois:
        return []

    rois_ord = sorted(rois, key=lambda r: -r[4])
    seleccionados = []

    while rois_ord:
        mejor = rois_ord.pop(0)
        seleccionados.append(mejor[:4])
        rois_ord = [r for r in rois_ord
                    if _iou(mejor, r) < umbral_iou]

    return seleccionados


def _iou(r1, r2) -> float:
    """Intersection over Union entre dos bboxes (x,y,w,h,...)."""
    x1 = max(r1[0], r2[0]); y1 = max(r1[1], r2[1])
    x2 = min(r1[0]+r1[2], r2[0]+r2[2])
    y2 = min(r1[1]+r1[3], r2[1]+r2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = r1[2]*r1[3] + r2[2]*r2[3] - inter
    return inter / (union + 1e-6)


# ════════════════════════════════════════════════════
#  PASO 2: CLASIFICACIÓN ZERO-SHOT
# ════════════════════════════════════════════════════

def clasificar_roi(
    roi_bgr    : np.ndarray,
    biblioteca : dict[str, BibliotecaClase],
    top_k      : int = TOP_K,
) -> list[tuple[str, float]]:
    """
    Codifica un ROI y lo compara con la biblioteca analítica
    usando similitud coseno sobre los vectores EFD + geometría.

    Retorna lista de (clase, score) ordenada por score desc.
    Tiempo típico: 3-8ms por ROI en CPU.
    """
    if not biblioteca:
        return [("desconocido", 0.0)]

    # Codificar ROI (rápido: solo EFD + geometría, sin analítico)
    resultado = codificar_forma(
        roi_bgr,
        incluir_efd       = True,
        incluir_geometria = True,
        incluir_analitico = False,   # desactivado para velocidad
    )

    if resultado.get("error") or len(resultado["vector"]) == 0:
        return [("desconocido", 0.0)]

    v_query = resultado["vector"]

    # Similitud coseno contra todos los vectores medios
    scores = []
    for nombre, clase_ref in biblioteca.items():
        if clase_ref.vector_medio is None:
            continue

        v_ref = clase_ref.vector_medio

        # Alinear dimensiones si difieren
        n = min(len(v_query), len(v_ref))
        if n < 10:
            continue

        sim = 1.0 - cosine(v_query[:n], v_ref[:n])
        sim = float(np.clip(sim, 0.0, 1.0))
        scores.append((nombre, sim))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k] if scores else [("desconocido", 0.0)]


# ════════════════════════════════════════════════════
#  PASO 3: CALIBRACIÓN DE CONFIANZA
# ════════════════════════════════════════════════════

def calibrar_confianza(
    similitud   : float,
    n_clases    : int,
    umbral      : float = UMBRAL_SIMILITUD,
) -> float:
    """
    Convierte la similitud coseno en una confianza calibrada.

    Aplica una sigmoide ajustada al número de clases:
    más clases → más difícil obtener alta similitud → se
    suaviza el umbral de aceptación.

    Retorna float en [0, 1]. Por debajo del umbral → 0.
    """
    if similitud < umbral:
        return 0.0

    # Escalar respecto al umbral y al número de clases
    pendiente = 10.0 / max(1, np.log(n_clases + 1))
    cal = 1.0 / (1.0 + np.exp(-pendiente * (similitud - (umbral + 0.15))))
    return float(np.clip(cal, 0.0, 1.0))


# ════════════════════════════════════════════════════
#  SISTEMA PRINCIPAL
# ════════════════════════════════════════════════════

class SistemaDeteccion:
    """
    Sistema completo de detección + clasificación zero-shot.

    Uso:
        sistema = SistemaDeteccion()

        # Registrar clases (offline, una vez)
        sistema.registrar_clase("circulo", "ref_circulo.png")
        sistema.registrar_clase("triangulo", "ref_triangulo.png")
        sistema.registrar_clase("A", "ref_letra_a.png")

        # Inferencia en tiempo real
        dets = sistema.detectar("imagen.png")
        sistema.visualizar("imagen.png", dets, "resultado.png")
    """

    def __init__(
        self,
        umbral_similitud : float = UMBRAL_SIMILITUD,
        escala_deteccion : float = ESCALA_RAPIDA,
        top_k            : int   = TOP_K,
        verboso          : bool  = False,
    ):
        self.biblioteca        = {}          # {nombre: BibliotecaClase}
        self.umbral_similitud  = umbral_similitud
        self.escala_deteccion  = escala_deteccion
        self.top_k             = top_k
        self.verboso           = verboso
        self._tiempos          = []          # historial de latencias

    # ── GESTIÓN DE LA BIBLIOTECA ──────────────────────

    def registrar_clase(
        self,
        nombre    : str,
        entrada   ,           # ruta str, array numpy, o lista de ambos
        verboso   : bool = True,
    ) -> "SistemaDeteccion":
        """
        Registra una clase en la biblioteca analítica.

        Acepta una sola imagen o una lista de imágenes de
        referencia. Con más referencias, la clasificación
        es más robusta.

        Retorna self para encadenar llamadas.
        """
        if nombre not in self.biblioteca:
            self.biblioteca[nombre] = BibliotecaClase(nombre=nombre,
                                                       vectores=[])

        entradas = entrada if isinstance(entrada, list) else [entrada]
        n_ok = 0

        for ent in entradas[:N_REFS_POR_CLASE]:
            try:
                res = codificar_forma(
                    ent,
                    incluir_efd       = True,
                    incluir_geometria = True,
                    incluir_analitico = False,
                )
                if not res.get("error") and len(res["vector"]) > 0:
                    self.biblioteca[nombre].vectores.append(res["vector"])
                    n_ok += 1
            except Exception as e:
                if verboso:
                    print(f"  ⚠ Error registrando '{nombre}': {e}")

        self.biblioteca[nombre].actualizar_medio()

        if verboso:
            dim = len(self.biblioteca[nombre].vector_medio) \
                  if self.biblioteca[nombre].vector_medio is not None else 0
            print(f"  ✓ Clase '{nombre}' registrada "
                  f"({n_ok} referencias, vector dim={dim})")

        return self

    def eliminar_clase(self, nombre: str) -> "SistemaDeteccion":
        """Elimina una clase de la biblioteca."""
        if nombre in self.biblioteca:
            del self.biblioteca[nombre]
            print(f"  Clase '{nombre}' eliminada")
        return self

    def clases(self) -> list[str]:
        """Lista de clases registradas."""
        return list(self.biblioteca.keys())

    # ── INFERENCIA PRINCIPAL ──────────────────────────

    def detectar(
        self,
        entrada,              # ruta str o array numpy BGR
        max_detecciones: int = 50,
    ) -> list[Deteccion]:
        """
        Pipeline completo: imagen → lista de Deteccion.

        Optimizado para < 100ms en CPU con imágenes típicas
        de documentos/pizarras (800px lado mayor).

        Pasos:
          1. Carga y redimensión (si necesario)
          2. Detección de ROIs por contornos (escala rápida)
          3. Para cada ROI: codificar + clasificar
          4. Filtrar por umbral de confianza
          5. Retornar detecciones ordenadas por confianza
        """
        t_inicio = time.perf_counter()

        # Cargar imagen
        if isinstance(entrada, str):
            img = cv2.imread(entrada)
            if img is None:
                raise FileNotFoundError(f"No encontrado: {entrada}")
        else:
            img = entrada.copy()

        # Redimensionar si es muy grande
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM_PROCESO:
            factor = MAX_DIM_PROCESO / max(h, w)
            img = cv2.resize(img, (int(w*factor), int(h*factor)),
                             interpolation=cv2.INTER_AREA)

        if self.verboso:
            print(f"\n  Imagen: {img.shape[1]}×{img.shape[0]} px  "
                  f"| Clases: {len(self.biblioteca)}")

        # ── Detección de ROIs ─────────────────────────
        t_det = time.perf_counter()
        rois = detectar_regiones(img, escala=self.escala_deteccion)
        t_det = (time.perf_counter() - t_det) * 1000

        if self.verboso:
            print(f"  ROIs detectados: {len(rois)}  ({t_det:.1f}ms)")

        # ── Clasificación de cada ROI ─────────────────
        detecciones = []
        n_clases    = len(self.biblioteca)

        for i, (x, y, bw, bh) in enumerate(rois[:max_detecciones]):
            t_roi = time.perf_counter()

            # Recortar ROI con margen de seguridad
            roi = img[max(0,y): min(img.shape[0], y+bh),
                      max(0,x): min(img.shape[1], x+bw)]

            if roi.size == 0 or min(roi.shape[:2]) < 8:
                continue

            candidatos = clasificar_roi(roi, self.biblioteca, self.top_k)
            t_roi = (time.perf_counter() - t_roi) * 1000

            if not candidatos:
                continue

            mejor_clase, mejor_sim = candidatos[0]
            confianza = calibrar_confianza(
                mejor_sim, n_clases, self.umbral_similitud)

            if confianza < 0.05:   # umbral mínimo de reporting
                continue

            cx = x + bw // 2
            cy = y + bh // 2

            det = Deteccion(
                clase      = mejor_clase if confianza >= 0.3 else "?",
                confianza  = confianza,
                bbox       = (x, y, bw, bh),
                centro     = (cx, cy),
                candidatos = candidatos,
                tiempo_ms  = t_roi,
            )
            detecciones.append(det)

            if self.verboso:
                print(f"    ROI {i+1}: [{mejor_clase}] "
                      f"sim={mejor_sim:.3f} conf={confianza:.3f} "
                      f"({t_roi:.1f}ms)")

        # Ordenar por confianza desc
        detecciones.sort(key=lambda d: -d.confianza)

        t_total = (time.perf_counter() - t_inicio) * 1000
        self._tiempos.append(t_total)

        if self.verboso or True:
            print(f"  ✓ {len(detecciones)} detecciones  "
                  f"| Total: {t_total:.1f}ms")

        return detecciones

    # ── VISUALIZACIÓN ─────────────────────────────────

    def visualizar(
        self,
        entrada,
        detecciones  : list[Deteccion],
        ruta_salida  : str = "detecciones.png",
        mostrar_topk : bool = True,
    ):
        """
        Dibuja los bounding boxes con clase y confianza.
        Guarda la imagen anotada.
        """
        if isinstance(entrada, str):
            img = cv2.imread(entrada)
        else:
            img = entrada.copy()

        # Redimensionar igual que en detectar()
        h_orig, w_orig = img.shape[:2]
        if max(h_orig, w_orig) > MAX_DIM_PROCESO:
            factor = MAX_DIM_PROCESO / max(h_orig, w_orig)
            img = cv2.resize(img,
                             (int(w_orig*factor), int(h_orig*factor)),
                             interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Paleta de colores por clase
        clases_unicas = list(set(d.clase for d in detecciones))
        cmap = plt.cm.get_cmap("tab20", max(len(clases_unicas), 1))
        color_map = {c: cmap(i)[:3] for i, c in enumerate(clases_unicas)}

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_rgb)

        for det in detecciones:
            x, y, bw, bh = det.bbox
            color = color_map.get(det.clase, (1, 0, 0))

            # Bounding box
            rect = patches.Rectangle(
                (x, y), bw, bh,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Etiqueta principal
            label = f"{det.clase}  {det.confianza:.2f}"
            ax.text(x, y - 4, label,
                    color="white", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=color, alpha=0.85))

            # Top-K candidatos (pequeño)
            if mostrar_topk and len(det.candidatos) > 1:
                for j, (cls, sc) in enumerate(det.candidatos[1:3], 1):
                    ax.text(x + bw + 3, y + j * 12,
                            f"{cls}: {sc:.2f}",
                            color="gray", fontsize=6)

        # Leyenda de clases
        handles = [patches.Patch(color=color_map[c], label=c)
                   for c in clases_unicas if c != "?"]
        if handles:
            ax.legend(handles=handles, loc="upper right",
                      fontsize=8, framealpha=0.8)

        latencia_media = np.mean(self._tiempos[-10:]) if self._tiempos else 0
        ax.set_title(
            f"{len(detecciones)} detecciones  |  "
            f"Clases: {len(self.biblioteca)}  |  "
            f"Latencia media: {latencia_media:.0f}ms",
            fontsize=10
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Resultado guardado: {ruta_salida}")

    # ── ESTADÍSTICAS ──────────────────────────────────

    def estadisticas(self) -> dict:
        """Resumen de rendimiento y configuración."""
        stats = {
            "n_clases"          : len(self.biblioteca),
            "clases"            : self.clases(),
            "refs_por_clase"    : {n: c.n_ejemplos
                                   for n, c in self.biblioteca.items()},
            "umbral_similitud"  : self.umbral_similitud,
        }
        if self._tiempos:
            stats["latencia_media_ms"] = float(np.mean(self._tiempos))
            stats["latencia_p95_ms"]   = float(np.percentile(self._tiempos, 95))
            stats["latencia_min_ms"]   = float(np.min(self._tiempos))
            stats["latencia_max_ms"]   = float(np.max(self._tiempos))
            stats["n_imagenes_procesadas"] = len(self._tiempos)
        return stats

    def imprimir_estadisticas(self):
        """Imprime las estadísticas en consola."""
        s = self.estadisticas()
        SEP = "─" * 50
        print(f"\n{SEP}\n  ESTADÍSTICAS DEL SISTEMA\n{SEP}")
        print(f"  Clases registradas : {s['n_clases']}")
        for nombre, n_refs in s["refs_por_clase"].items():
            print(f"    · {nombre:20s} {n_refs} referencia(s)")
        if "latencia_media_ms" in s:
            print(f"\n  Latencia media : {s['latencia_media_ms']:.1f}ms")
            print(f"  Latencia p95   : {s['latencia_p95_ms']:.1f}ms")
            print(f"  Rango          : [{s['latencia_min_ms']:.1f}, "
                  f"{s['latencia_max_ms']:.1f}]ms")
            print(f"  Imágenes proc. : {s['n_imagenes_procesadas']}")
        print(SEP)

    # ── PERSISTENCIA ──────────────────────────────────

    def guardar(self, ruta: str = "biblioteca.npz"):
        """
        Guarda la biblioteca de clases en disco.
        Permite reutilizarla sin volver a registrar.
        """
        data = {}
        meta = {}
        for nombre, clase in self.biblioteca.items():
            if clase.vector_medio is not None:
                key = f"clase__{nombre}"
                data[key] = clase.vector_medio
                meta[nombre] = clase.n_ejemplos

        np.savez_compressed(ruta, **data)
        with open(ruta.replace(".npz", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Biblioteca guardada: {ruta}  ({len(data)} clases)")

    def cargar(self, ruta: str = "biblioteca.npz") -> "SistemaDeteccion":
        """
        Carga una biblioteca previamente guardada.
        Las clases cargadas se añaden a las ya existentes.
        """
        archivo = np.load(ruta)
        meta_ruta = ruta.replace(".npz", "_meta.json")
        meta = {}
        if os.path.exists(meta_ruta):
            with open(meta_ruta) as f:
                meta = json.load(f)

        for key in archivo.files:
            if key.startswith("clase__"):
                nombre = key[7:]
                if nombre not in self.biblioteca:
                    self.biblioteca[nombre] = BibliotecaClase(
                        nombre=nombre, vectores=[])
                self.biblioteca[nombre].vector_medio = archivo[key]
                self.biblioteca[nombre].n_ejemplos   = meta.get(nombre, 1)

        print(f"  Biblioteca cargada: {len(archivo.files)} clases desde {ruta}")
        return self


# ════════════════════════════════════════════════════
#  EVALUACIÓN (cuando hay datos etiquetados)
# ════════════════════════════════════════════════════

def evaluar(
    sistema     : SistemaDeteccion,
    rutas       : list[str],
    etiquetas   : list[str],
    umbral_iou  : float = 0.5,
) -> dict:
    """
    Evalúa el sistema sobre un conjunto de imágenes etiquetadas.

    Para cada imagen espera UNA clase correcta (imagen completa).
    Útil para medir precisión de clasificación cuando no tienes
    anotaciones de bounding box.

    Retorna dict con accuracy, precision por clase, latencias.
    """
    correctas = 0
    total     = 0
    por_clase = {}

    for ruta, etiqueta in zip(rutas, etiquetas):
        dets = sistema.detectar(ruta)

        pred = dets[0].clase if dets else "ninguna"
        ok   = (pred == etiqueta)

        correctas += int(ok)
        total     += 1

        if etiqueta not in por_clase:
            por_clase[etiqueta] = {"ok": 0, "total": 0}
        por_clase[etiqueta]["total"] += 1
        por_clase[etiqueta]["ok"]    += int(ok)

    accuracy = correctas / total if total > 0 else 0.0
    precision_por_clase = {
        c: v["ok"] / v["total"]
        for c, v in por_clase.items()
    }

    resultado = {
        "accuracy"           : accuracy,
        "precision_por_clase": precision_por_clase,
        "n_correctas"        : correctas,
        "n_total"            : total,
        **sistema.estadisticas(),
    }

    print(f"\n  Accuracy: {accuracy:.3f}  ({correctas}/{total})")
    for c, p in sorted(precision_por_clase.items()):
        print(f"    {c:20s}: {p:.3f}")

    return resultado


# ════════════════════════════════════════════════════
#  EJEMPLO DE USO
# ════════════════════════════════════════════════════
if __name__ == "__main__":
    import glob

    # ── Ejemplo A: uso básico ─────────────────────────
    sistema = SistemaDeteccion(
        umbral_similitud = 0.45,
        verboso          = True,
    )

    # Registrar clases con imágenes de referencia
    # (1 imagen mínimo, hasta 5 para más robustez)
    sistema.registrar_clase("circulo",   "refs/circulo.png")
    sistema.registrar_clase("triangulo", "refs/triangulo.png")
    sistema.registrar_clase("cuadrado",  "refs/cuadrado.png")
    sistema.registrar_clase("A",         "refs/letra_a.png")
    sistema.registrar_clase("B",         "refs/letra_b.png")

    # Guardar biblioteca para no tener que re-registrar
    sistema.guardar("mi_biblioteca.npz")

    # Detectar en una imagen nueva
    detecciones = sistema.detectar("documento.png")

    # Ver resultados
    for d in detecciones:
        print(d)

    # Guardar imagen anotada
    sistema.visualizar("documento.png", detecciones, "resultado.png")

    # Estadísticas de rendimiento
    sistema.imprimir_estadisticas()

    # ── Ejemplo B: añadir clase nueva sin reentrenar ──
    # sistema.cargar("mi_biblioteca.npz")
    # sistema.registrar_clase("estrella", "refs/estrella.png")
    # sistema.guardar("mi_biblioteca_v2.npz")

    # ── Ejemplo C: evaluación con datos etiquetados ───
    # rutas     = glob.glob("test/*.png")
    # etiquetas = [os.path.basename(r).split("_")[0] for r in rutas]
    # metricas  = evaluar(sistema, rutas, etiquetas)
