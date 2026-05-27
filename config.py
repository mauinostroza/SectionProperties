"""Configuration constants for SectionAnalyzer Pro."""

SECTION_TYPES = {
    "Rectangular": {"icon": "📦", "params": ["b", "h"]},
    "Circular": {"icon": "🔵", "params": ["d"]},
    "T": {"icon": "⫘", "params": ["b_f", "h_f", "b_w", "h_w"]},
    "I": {"icon": "⧉", "params": ["b_f", "h_f", "b_w", "h_w", "t_f"]},
    "L": {"icon": "📐", "params": ["b", "h", "t"]},
    "Cajón": {"icon": "⬜", "params": ["b_ext", "h_ext", "b_int", "h_int"]},
    "Compuesta": {"icon": "🔗", "params": []},
    "Poligonal": {"icon": "⬡", "params": []},
}

MATERIAL_PRESETS = {
    "Concrete": {
        "H-20": {"fc": 20, "Ec": 22000, "gamma": 24.0, "nu": 0.2},
        "H-25": {"fc": 25, "Ec": 25000, "gamma": 24.0, "nu": 0.2},
        "H-30": {"fc": 30, "Ec": 27000, "gamma": 24.0, "nu": 0.2},
        "H-35": {"fc": 35, "Ec": 29000, "gamma": 24.0, "nu": 0.2},
        "H-40": {"fc": 40, "Ec": 30000, "gamma": 24.0, "nu": 0.2},
        "H-50": {"fc": 50, "Ec": 32000, "gamma": 24.0, "nu": 0.2},
    },
    "Steel": {
        "B400B": {"fy": 400, "Es": 200000, "gamma": 78.5},
        "B500B": {"fy": 500, "Es": 200000, "gamma": 78.5},
        "A615-60": {"fy": 420, "Es": 200000, "gamma": 78.5},
        "A706-60": {"fy": 420, "Es": 200000, "gamma": 78.5},
        "A36": {"fy": 250, "Es": 200000, "gamma": 78.5},
    },
}

UNIT_SYSTEMS = {
    "kN, m": {"force": 1.0, "length": 1.0, "stress": 1.0, "label": "kN, m"},
    "kgf, cm": {"force": 0.00980665, "length": 0.01, "stress": 0.0980665, "label": "kgf, cm"},
    "lbf, in": {"force": 0.00444822, "length": 0.0254, "stress": 0.00689476, "label": "lbf, in"},
}

DESIGN_CODES = ["ACI 318-19", "Eurocode 2", "EHE-08", "NSR-10", "NCh433"]

DEFAULT_REBAR_DIAMETERS = [8, 10, 12, 16, 18, 20, 22, 25, 28, 32, 36, 40]

DEFAULT_REBAR_LAYERS = ["Ext. inf", "Ext. sup", "Media", "Estribo"]

DEFAULT_REBAR_MATERIALS = ["B500B", "B400B", "A615-60", "A706-60", "A36"]

CONCRETE_MODELS = [
    "Parabólica-rectangular (EC2)",
    "Bilineal (ACI)",
    "Popovics",
    "Personalizada",
]

STEEL_MODELS = [
    "Elastoplástica perfecta",
    "Con endurecimiento",
    "Ramberg-Osgood",
]

SOLVER_TYPES = ["Directo (Cholesky)", "Iterativo (CG)", "Auto"]

LOAD_HISTORIES = ["Monotónica", "Cíclica simétrica", "Cíclica asimétrica"]
