# 📐 Análisis de Secciones Transversales

App [Streamlit](https://streamlit.io) para calcular propiedades geométricas, plásticas, tensiones, diagramas de interacción P-M y curva M-φ de secciones transversales de acero, hormigón armado y compuestas.

**Motor:** `sectionproperties` v3.x (FEM) · Diagrama P-M: Bloque de Whitney (ACI 318 / NCh430) · Curva M-φ: Hognestad

## 🚀 Publicada en

[https://sectionproperties.streamlit.app](https://sectionproperties.streamlit.app)

## 🧰 Instalación local

```bash
git clone https://github.com/mauinostroza/SectionProperties.git
cd SectionProperties
pip install -r requirements.txt
streamlit run app.py
```

## 🏗️ Secciones soportadas

| Categoría | Tipos |
|-----------|-------|
| 🔩 Acero | Perfil I (doble T), Canal (C/UPN), Ángulo (L), Perfil T, RHS, CHS |
| 🏗️ Hormigón | Rectangular, Circular, Viga T, Doble T (I-H) |
| 🔧 Compuesta | Viga mixta (perfil acero + losa HA) |

## 📊 Análisis incluidos

- **Geométricas:** A, Ixx, Iyy, Ixy, I₁₁, I₂₂, Z, r, Q
- **Alabeo:** J, Iw, centro de corte, As (área de corte)
- **Plásticas:** Zpx, Zpy, factores de forma, centroide plástico
- **Tensiones:** σ axial, σ flexión, τ corte, τ torsión, Von Mises
- **Diagrama P-M:** Interacción axial-momento (ACI 318 / NCh430)
- **Curva M-φ:** Momento-curvatura con ductilidad μφ
- **🌐 P-Mx-My 3D:** Superficie de interacción 3D interactiva (Plotly)

## 🌐 Diagrama 3D P-Mx-My

Barre el eje neutro 0°→360° y construye la superficie de falla 3D con verificación ACI 318:

- Rotación interactiva con el mouse (Plotly)
- Punto de demanda (P, Mx, My) marcado
- Verifica si la demanda cumple o no con la envolvente φRn

## 📤 Exportación

- Tabla resumen a **Excel** (.xlsx)
- Curva M-φ a **CSV**

## ⚠️ Nota

El usuario es responsable de verificar y validar todos los resultados. Esta herramienta es de apoyo al diseño, no reemplaza el criterio ingenieril ni los cálculos detallados según normativa.
