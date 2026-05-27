# SectionAnalyzer Pro 🔲

Aplicación Streamlit profesional para análisis de secciones transversales de hormigón armado, acero y compuestas.

**Stack:** sectionproperties + concreteproperties + Plotly + pandas

## Arquitectura multi-módulo

```
├── app.py              # Entry point, layout 3 columnas
├── config.py           # Constantes, unidades, materiales predefinidos
├── state_manager.py    # Session state centralizado
├── modules/
│   ├── section_module.py    # Módulo Sección (biblioteca, geometría, refuerzo, materiales, malla)
│   ├── analysis_module.py   # Módulo Análisis (propiedades, M-N, M-κ, tensiones, SLS)
│   └── batch_module.py      # Módulo Batch (procesamiento masivo)
├── components/
│   ├── header.py            # Barra superior persistente
│   ├── sidebar_nav.py       # Navegación lateral
│   ├── geometry_canvas.py   # Canvas 2D con Plotly
│   ├── rebar_editor.py      # Editor de refuerzo tipo Excel
│   ├── mn_diagram.py        # Diagrama M-N
│   ├── mk_diagram.py        # Diagrama M-κ
│   ├── stress_plot.py       # Tensiones y deformaciones
│   └── properties_panel.py  # Wrapper de panel derecho
├── utils/
│   ├── geometry_factory.py  # Factory de secciones sectionproperties
│   ├── material_library.py  # Catálogo de materiales
│   ├── excel_handler.py     # Import/export Excel
│   └── plotly_themes.py     # Temas de color consistentes
└── data/
    └── default_sections.json
```

## Plan de rediseño

Ver `plan_rediseno.md` para la arquitectura completa y roadmap.
