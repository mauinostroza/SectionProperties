#!/usr/bin/env python3
"""Test rápido de la superficie 3D P-Mx-My."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Extraer solo las funciones necesarias
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")

# Copiar funciones necesarias
from app import (
    distribuir_barras, Ab_circ, rc_pm3d_surface, _clip_rect_halfplane,
    _polygon_area, _polygon_centroid, _build_pm3d_grid
)

b, h = 300, 500
fc, fy, Es = 25, 420, 200000
n_bars, d_bar, cover = 6, 20, 40

bar_y, bar_x = distribuir_barras(b, h, n_bars, cover, d_bar)
Ab = Ab_circ(d_bar)

print(f"As = {n_bars * Ab:.0f} mm²")
print(f"Barras (x, y): {list(zip(map(int, bar_x), map(int, bar_y)))}")

# Prueba del clipping
n = np.array([0.0, 1.0])  # θ=0°
poly = _clip_rect_halfplane(b, h, n, h - 100)  # NA a 100mm del tope
area = _polygon_area(poly)
cent = _polygon_centroid(poly)
print(f"θ=0° NA@100mm: área={area:.0f} mm², centroide=({cent[0]:.1f}, {cent[1]:.1f})")

result = rc_pm3d_surface(b, h, fc, fy, Es, bar_x, bar_y, Ab,
                          phi_col=0.65, n_theta=12, n_c=30)
print(f"\nP0 = {result['P0']:.1f} kN")
print(f"φP0 = {result['phiP0']:.1f} kN")
if result["Mx"].size > 0:
    print(f"Mx grid: {result['Mx'].shape}")
    print(f"Rango Mx: {np.nanmin(result['Mx']):.2f} a {np.nanmax(result['Mx']):.2f} kN·m")
    print(f"Rango My: {np.nanmin(result['My']):.2f} a {np.nanmax(result['My']):.2f} kN·m")
    print(f"Rango P:  {np.nanmin(result['P']):.2f} a {np.nanmax(result['P']):.2f} kN")
    print(f"Puntos curva: {len(result['Mx_d'])}")
print("✅ Superficie 3D OK")
