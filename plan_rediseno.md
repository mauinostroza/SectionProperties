# Plan de Rediseño Completo - SectionAnalyzer Pro

## 1. Visión General

Transformar la aplicación Streamlit actual basada en `sectionproperties`/`concrete-properties` en una herramienta profesional de análisis de secciones transversales de hormigón y compuestas, con UI moderna inspirada en Engissol, SkyCiv e IDEA StatiCa.

**Objetivos clave:**
- Canvas central dominante (55-60% del ancho) para visualización 2D/3D y gráficos.
- Panel lateral derecho (30-35%) para propiedades y configuración.
- Navegación por 3 módulos principales: **Sección**, **Análisis** y **Batch**.
- Ingreso masivo de datos mediante editor tipo Excel (`st.data_editor`) y copiar/pegar.
- Actualización en tiempo real de la geometría y resultados.
- Opciones avanzadas ocultas bajo expanders y modales.

---

## 2. Arquitectura de Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HEADER (50px)  [Logo]  [Proyecto]  [Unidades]  [💾]  [📄]  [⚙️]            │
├──────────┬──────────────────────────────────────────┬───────────────────────┤
│          │                                          │                       │
│  NAV     │         CANVAS CENTRAL                   │   PANEL DE            │
│  LATERAL │         (55-60% ancho)                   │   PROPIEDADES         │
│  (80px)  │                                          │   (30% ancho)         │
│          │    • Visualización 2D/3D               │   • Formularios       │
│  [🔲]    │    • Gráficos interactivos               │   • Tablas            │
│  [📊]    │    • Diagramas M-N, M-κ                  │   • Sliders           │
│  [⚡]    │    • Deformadas y tensiones              │   • Selectores        │
│          │                                          │   • Botones de acción │
├──────────┴──────────────────────────────────────────┴───────────────────────┤
│  FOOTER (30px)  [Estado: Mesh OK | 1,240 elementos]  [⏱️ Cálculo: 0.4s]    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proporciones recomendadas

| Zona | % Ancho | Rol |
|------|---------|-----|
| Navegación lateral | 5-8% | Iconos grandes, 3 items. Colapsable en móviles. |
| Canvas central | 55-60% | Visualización principal. Nunca comprimido. |
| Panel de propiedades | 30-35% | Forms, tablas, selectores. Grupos con bordes. |
| Header | 50-60px fijo | Contexto global siempre visible. |
| Footer | 30px fijo | Estado del sistema, feedback no intrusivo. |

---

## 3. Módulos y Sub-pestañas

### Módulo 1: 🔲 Sección Transversal

**Objetivo:** Definir, elegir o importar la geometría, materiales y refuerzo.

| Sub-pestaña | Canvas Central | Panel Derecho (Propiedades) |
|-------------|----------------|-----------------------------|
| **Biblioteca** | Galería de secciones predefinidas (rectangular, T, circular, I, perfiles de acero AISC/Euro, compuestas). Vista previa en miniatura. | Filtros por tipo, norma, material. Botón "Usar esta sección". |
| **Geometría** | Canvas 2D interactivo con grid, zoom, snap. Dibujo de polígonos, círculos, rectángulos, agujeros. | Coordenadas de puntos, dimensiones (b, h, r, t), operaciones booleanas (unión, resta), offset de perímetro. |
| **Refuerzo** | Visualización de barras sobre la sección. Diferentes colores por diámetro. | Diámetro, cantidad, posición (x,y), capas. **Entrada masiva**: botón "Importar desde Excel/CSV" o "Copiar desde tabla". |
| **Materiales** | Leyenda de colores por material en la sección. | f'c, fy, E, ν, γ. Selector de curvas: bilineal, parabólica, personalizada (hormigón confinado). |
| **Malla** | Preview del mesh FEM con densidad visualizada. | Tamaño de elemento, ángulo máximo, calidad. *Opción avanzada*: tipo de solver (directo/iterativo). |

**Ingreso masivo de refuerzo:**
- Botón **"Editor tipo Excel"** que abre `st.data_editor()` con columnas: `Barra #`, `Diámetro (mm)`, `X (mm)`, `Y (mm)`, `Material`, `Capa`.
- Opciones de **pegar desde Excel/CSV** directamente.
- Botones de **replicar patrón**: línea de barras, circular, rectangular.

---

### Módulo 2: 📊 Análisis y Resultados

**Objetivo:** Ejecutar cálculos y visualizar todos los resultados gráficos.

| Sub-pestaña | Canvas Central | Panel Derecho (Configuración) |
|-------------|----------------|-------------------------------|
| **Propiedades Geométricas** | Tabla resumen + elipse de inercia + centroides (C, S, P) visualizados. | Seleccionar eje (global/centroidal/principal). Botón "Exportar a LaTeX/PDF". |
| **Diagrama M-N** | Gráfico interactivo (Plotly) con curvas para diferentes cuantías (ρ=1%, 2%, 3%) o ejes (x-x, y-y). | Cuantías a evaluar, ángulo de inclinación, casos de carga superpuestos (puntos de diseño). |
| **Momento-Curvatura (M-κ)** | Curva M vs κ con punto seleccionable. Al hacer click, se actualiza la sub-pestaña "Deformada". | Nivel de carga axial (P), historia de carga (monotónica/cíclica). Opción de bilinealización automática. |
| **Deformada / Tensiones** | Split view: arriba la sección deformada con campo de tensiones (contour plot), abajo distribución de deformaciones lineal. | Caso de carga seleccionado, escala de deformación, tipo de visualización (σxx, σvm, ε). Tabla de fuerzas en barras. |
| **Estado Límite** | Mapa de color de ratios de capacidad sobre la sección. | Código de diseño (ACI, EC2, etc.), factores de seguridad, verificación ULS/SLS. |

**Interacción entre sub-pestañas:**
Al hacer clic en un punto del diagrama M-N, la sub-pestaña "Deformada" se actualiza automáticamente mostrando la distribución de tensiones para ese punto exacto.

---

### Módulo 3: ⚡ Datos Masivos (Batch)

**Objetivo:** Procesar múltiples secciones o múltiples casos de carga en una sola ejecución.

| Área | Funcionalidad |
|------|---------------|
| **Tabla maestra** | `st.data_editor()` con columnas: ID, Tipo de sección, b, h, recubrimiento, fc, fy, As total, P aplicado, Mx, My. Permite copiar/pegar desde Excel. |
| **Importación** | Drag & drop de `.xlsx` o `.csv`. Mapeo automático de columnas con validación. |
| **Ejecución** | Barra de progreso con cálculo paralelo. Resultados en tiempo real. |
| **Resultados consolidados** | Tabla de resultados: ID, φMn, φPn, ratio, capacidad, estado (OK/Fail). Filtros y búsqueda. |
| **Exportación** | Descargar resultados completos en Excel con múltiples hojas: Inputs, Propiedades, M-N, M-κ, Tensiones. |

---

## 4. Jerarquía de Opciones (Visibilidad)

| Nivel | Elemento de UI | Ubicación |
|-------|----------------|-----------|
| **Primario** (siempre visible) | Tipo de sección, dimensiones básicas, materiales básicos (fc, fy), refuerzo principal. | Panel derecho, secciones expandidas por defecto. |
| **Secundario** (un clic away) | Curvas tensión-deformación personalizadas, parámetros de mesh avanzados, códigos de diseño alternativos, unidades personalizadas. | `st.expander("Opciones avanzadas")` o botón ⚙️ que abre un drawer/popover. |
| **Terciario** (oculto por defecto) | Solver iterativo vs directo, tolerancias de convergencia, exportar a formatos específicos (DXF, Rhino .3dm), temas de color. | Dentro de "Configuración avanzada", o en un modal de Settings global. |

---

## 5. Stack Tecnológico Recomendado

| Componente | Librería | Justificación |
|------------|----------|---------------|
| Framework UI | `streamlit` | Base actual. Usar `layout="wide"`, `st.columns`, `st.tabs`, `st.expander`, `st.data_editor`. |
| Visualización interactiva | `plotly` | Zoom, pan, hover, selección de puntos. Mejor que matplotlib estático. |
| Backend secciones | `sectionproperties` + `concreteproperties` | Cálculo de propiedades geométricas y análisis de hormigón armado. |
| Tablas tipo Excel | `st.data_editor` (nativo Streamlit) | Copiar/pegar celdas, edición inline, añadir/eliminar filas dinámicamente. |
| Importación batch | `pandas` + `openpyxl` | Lectura de Excel/CSV, validación de datos, procesamiento masivo. |
| Exportación | `pandas` + `xlsxwriter` / `reportlab` | Generación de Excel multi-hoja y PDFs. |
| Caché de cálculos | `st.cache_data` | Evitar recalcular secciones idénticas. Mejora rendimiento. |
| Estado de sesión | `st.session_state` | Persistencia de la sección activa, resultados y configuración entre pestañas. |

---

## 6. Estructura de Archivos Propuesta

```
sectionanalyzer-pro/
├── app.py                    # Punto de entrada, layout global, navegación
├── config.py                 # Unidades, códigos de diseño, constantes
├── state_manager.py            # Gestión de st.session_state (sección activa, resultados)
├── modules/
│   ├── section_module.py       # Lógica del Módulo 1: Sección
│   ├── analysis_module.py      # Lógica del Módulo 2: Análisis
│   └── batch_module.py         # Lógica del Módulo 3: Batch
├── components/
│   ├── header.py               # Barra superior persistente
│   ├── sidebar_nav.py          # Navegación lateral por iconos
│   ├── properties_panel.py     # Wrapper del panel derecho
│   ├── geometry_canvas.py      # Renderizado 2D con plotly
│   ├── rebar_editor.py         # Editor tipo Excel de refuerzo
│   ├── mn_diagram.py           # Cálculo y gráfico M-N
│   ├── mk_diagram.py           # Cálculo y gráfico M-κ
│   └── stress_plot.py          # Contour plots de tensiones
├── utils/
│   ├── geometry_factory.py     # Factory de secciones predefinidas
│   ├── material_library.py     # Catálogo de materiales (ACI, EC2)
│   ├── excel_handler.py        # Importar/exportar Excel
│   └── plotly_themes.py        # Temas de color consistentes
├── data/
│   └── default_sections.json   # Secciones predefinidas de ejemplo
└── requirements.txt
```

---

## 7. Flujo de Trabajo del Usuario

1. **Llega a la app** → Ve directamente el Módulo "Sección" con una sección rectangular por defecto y el canvas central mostrándola.
2. **Define geometría** → Cambia b y h en el panel derecho, ve la actualización en tiempo real.
3. **Añade refuerzo** → Clic en sub-pestaña "Refuerzo", usa el editor tipo Excel para pegar 20 barras desde su planilla.
4. **Cambia a Análisis** → Clic en navegación lateral. Ve automáticamente las Propiedades Geométricas calculadas.
5. **Genera M-N** → Clic en sub-pestaña "Diagrama M-N", ajusta cuantías en panel derecho, clic en "Calcular".
6. **Explora interactivamente** → Clic en un punto de la curva M-N → cambia a "Deformada" y ve la distribución de tensiones para ese estado.
7. **Batch** → Si tiene 50 pilares, va a "Datos Masivos", pega la tabla, ejecuta, descarga Excel con todos los checks.

---

## 8. Notas de Implementación Críticas

### 8.1 Estado Global (Session State)
Es fundamental mantener en `st.session_state`:
- `active_section`: Objeto `ConcreteSection` o `Section` actual.
- `rebar_table`: DataFrame con el refuerzo.
- `materials`: Dict de materiales (hormigón, acero).
- `analysis_results`: Dict con resultados de M-N, M-κ, propiedades.
- `batch_data`: DataFrame con datos masivos.
- `settings`: Configuración de unidades, código, tema.

### 8.2 Rendimiento
- Usar `st.cache_data` para el cálculo de propiedades geométricas y diagramas M-N cuando los inputs no cambien.
- El cálculo de M-κ es costoso; mostrar spinner y calcular bajo demanda (botón explícito), no en cada cambio de parámetro.
- Para batch, usar `st.progress` y procesamiento por chunks si son >100 secciones.

### 8.3 Responsive
- En pantallas <1200px, el panel de propiedades debe pasar a un drawer lateral (`st.sidebar` o `st.popover`) para no comprimir el canvas.
- La navegación lateral puede colapsarse a iconos sin texto en tablets.

### 8.4 Validación de Datos
- Validar que el refuerzo no quede fuera de la sección (cover mínimo).
- Validar que fc > 0, fy > 0, dimensiones > 0 antes de calcular.
- Mostrar errores con `st.error` en el panel derecho, nunca en el canvas.

---

## 9. Roadmap Sugerido de Implementación

| Fase | Duración estimada | Entregable |
|------|-------------------|------------|
| **Fase 1: Layout y Navegación** | 2-3 días | Estructura de 3 columnas, header, footer, navegación lateral funcional entre módulos. |
| **Fase 2: Módulo Sección** | 4-5 días | Biblioteca de secciones, canvas 2D básico, editor de refuerzo con data_editor, importación CSV. |
| **Fase 3: Módulo Análisis** | 5-7 días | Integración con sectionproperties/concreteproperties, gráficos M-N y M-κ interactivos, deformadas y tensiones. |
| **Fase 4: Módulo Batch** | 3-4 días | Tabla maestra, procesamiento masivo, exportación Excel multi-hoja. |
| **Fase 5: Polish** | 2-3 días | Opciones avanzadas, temas de color, exportación PDF/LaTeX, optimización de rendimiento. |

**Total estimado:** 16-22 días de desarrollo por un ingeniero con experiencia en Streamlit y Python estructural.

---

*Documento generado el 2026-05-27 para el proyecto SectionAnalyzer Pro.*
