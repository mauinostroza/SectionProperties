"""
app.py  ·  Análisis de Secciones Transversales
================================================
App Streamlit usando sectionproperties v3.x (FEM)
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from io import BytesIO
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sectionproperties.pre.library import (
    i_section, channel_section, angle_section,
    rectangular_section, circular_section,
    circular_hollow_section, rectangular_hollow_section,
)
from sectionproperties.pre import Geometry, CompoundGeometry, Material
from sectionproperties.analysis import Section

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Secciones Transversales",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a,#1a3a5f);}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,[data-testid="stSidebar"] div{color:#cfd8dc!important;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#82b1ff!important;}
[data-testid="stSidebar"] hr{border-color:#37474f!important;}
.app-header{background:linear-gradient(135deg,#1565C0,#0288D1);
            padding:20px 28px;border-radius:12px;margin-bottom:20px;}
.app-header h1{color:white!important;margin:0;font-size:1.7rem;}
.app-header p{color:#B3E5FC!important;margin:6px 0 0;font-size:.87rem;}
.pcard{background:#fff;border-left:4px solid #1565C0;border-radius:4px 8px 8px 4px;
       padding:8px 14px;margin:3px 0;box-shadow:0 1px 4px rgba(0,0,0,.07);}
.pcard .pn{font-size:.71rem;font-weight:700;text-transform:uppercase;
           letter-spacing:.5px;color:#607D8B;}
.pcard .pv{font-size:1.05rem;font-weight:700;color:#1a3a5c;}
.pcard .pu{font-size:.77rem;color:#90A4AE;margin-left:5px;}
.warn-box{background:#FFF8E1;border-left:4px solid #FFC107;
          padding:10px 14px;border-radius:4px;margin:8px 0;}
.ok-box{background:#E8F5E9;border-left:4px solid #43A047;
        padding:10px 14px;border-radius:4px;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <h1>📐 Análisis de Secciones Transversales</h1>
  <p>Propiedades geométricas · Alabeo · Plásticas · Tensiones ·
     Diagrama P-M · Curva M-φ &nbsp;|&nbsp;
     Motor: <code>sectionproperties</code> v3.x (FEM)</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────
def pcard(name, value, unit="", fmt=".5g"):
    vs = f"{value:{fmt}}" if isinstance(value, (float, np.floating, int)) else str(value)
    st.markdown(
        f'<div class="pcard"><div class="pn">{name}</div>'
        f'<div class="pv">{vs}<span class="pu">{unit}</span></div></div>',
        unsafe_allow_html=True,
    )

def show_fig(fig, caption=""):
    st.pyplot(fig, use_container_width=True)
    if caption: st.caption(caption)
    plt.close(fig)

def fmt_si(v):
    try: return f"{float(v):.5g}"
    except: return str(v)

def mytrapz(y, x):
    try:    return np.trapezoid(y, x)
    except: return np.trapz(y, x)

def Ab_circ(d_bar): return np.pi * (d_bar / 2) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRÍAS AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────
def make_t_steel(d, b, t_f, t_w):
    flange = rectangular_section(d=t_f, b=b)
    web = (rectangular_section(d=d-t_f, b=t_w)
           .align_center(align_to=flange)
           .align_to(other=flange, on="bottom"))
    return flange + web

def make_concrete_T(b_f, d_f, b_w, d_w):
    web = rectangular_section(d=d_w, b=b_w)
    flange = (rectangular_section(d=d_f, b=b_f)
              .align_center(align_to=web)
              .align_to(other=web, on="top"))
    return web + flange

def make_double_T(b_f, d_f, b_w, d_w, b_fb, d_fb):
    web = rectangular_section(d=d_w, b=b_w)
    top = (rectangular_section(d=d_f, b=b_f)
           .align_center(align_to=web).align_to(other=web, on="top"))
    bot = (rectangular_section(d=d_fb, b=b_fb)
           .align_center(align_to=web).align_to(other=web, on="bottom"))
    return web + top + bot


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUCIÓN DE BARRAS
# ─────────────────────────────────────────────────────────────────────────────
def distribuir_barras(b_rc, h_rc, n_bars, cover, d_bar):
    """Retorna (bar_y, bar_x) — posiciones desde la BASE y el borde IZQ (mm)."""
    r = cover + d_bar / 2
    bars_y, bars_x = [], []
    if n_bars <= 4:
        cy = [r, r, h_rc - r, h_rc - r]
        cx = [r, b_rc - r, r, b_rc - r]
        for i in range(n_bars):
            bars_y.append(cy[i]); bars_x.append(cx[i])
    else:
        cy = [r, r, h_rc - r, h_rc - r]
        cx = [r, b_rc - r, r, b_rc - r]
        for y, x in zip(cy, cx):
            bars_y.append(y); bars_x.append(x)
        extra = n_bars - 4
        n_bot = extra // 2; n_top = extra - n_bot
        if n_bot > 0:
            for x in np.linspace(r, b_rc - r, n_bot + 2)[1:-1]:
                bars_y.append(r); bars_x.append(float(x))
        if n_top > 0:
            for x in np.linspace(r, b_rc - r, n_top + 2)[1:-1]:
                bars_y.append(h_rc - r); bars_x.append(float(x))
    return np.array(bars_y), np.array(bars_x)


# ─────────────────────────────────────────────────────────────────────────────
# DIBUJO SECCIÓN HA
# ─────────────────────────────────────────────────────────────────────────────
def dibujar_seccion_ha(ax, b, h, bar_y, bar_x, d_bar, cover, titulo="", cx=None, cy=None):
    ax.add_patch(patches.Rectangle((0,0), b, h, lw=1.5,
                                    edgecolor="#333", facecolor="#e0e0e0"))
    r = cover
    ax.add_patch(patches.Rectangle((r,r), b-2*r, h-2*r, lw=1,
                                    edgecolor="#795548", facecolor="none", ls="--"))
    for yi, xi in zip(bar_y, bar_x):
        ax.add_patch(patches.Circle((xi, yi), d_bar/2,
                                     facecolor="#1565C0", edgecolor="white", lw=0.5))
    if cx is not None:
        ax.plot(cx, cy, "r+", ms=14, mew=2.5, zorder=5, label="Centroide")
    ax.set_xlim(-b*0.05, b*1.05); ax.set_ylim(-h*0.05, h*1.05)
    ax.set_aspect("equal"); ax.set_title(titulo, fontsize=9)
    ax.set_xlabel("x (mm)", fontsize=8); ax.set_ylabel("y (mm)", fontsize=8)
    ax.grid(True, alpha=0.2, lw=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAMA P-M  (Whitney / ACI 318)
# ─────────────────────────────────────────────────────────────────────────────
def rc_pm_diagram(b, h, fc, fy, Es, bar_y, Ab_each, phi_col=0.65):
    ecu   = 0.003
    beta1 = max(0.65, 0.85 - 0.05*(fc-28)/7) if fc > 28 else 0.85
    bar_d = h - np.asarray(bar_y, dtype=float)
    As_arr = np.full(len(bar_d), Ab_each)

    def forces(c):
        c = max(c, 1e-6)
        a = min(beta1*c, h)
        Cc = 0.85*fc*a*b
        eps_s = ecu*(c - bar_d)/c
        fs    = np.clip(eps_s*Es, -fy, fy)
        fs_net = fs.copy()
        fs_net[bar_d <= a] -= 0.85*fc
        Pn = Cc + np.sum(fs_net*As_arr)
        Mn = Cc*(h/2 - a/2) + np.sum(fs_net*As_arr*(h/2 - bar_d))
        return Pn, Mn

    c_vals = np.concatenate([np.linspace(1e-4, h, 300),
                              np.linspace(h, 8*h, 80), [30*h]])
    Pns, Mns = zip(*[forces(c) for c in c_vals])
    Pns = np.array(Pns)/1e3; Mns = np.abs(Mns)/1e6

    As_tot = Ab_each * len(bar_y)
    P0 = (0.85*fc*(b*h - As_tot) + As_tot*fy)/1e3
    Pns = np.clip(Pns, None, P0)
    Pns = np.append(Pns, -As_tot*fy/1e3); Mns = np.append(Mns, 0.)

    idx = np.argsort(Pns)
    Pn  = Pns[idx]; Mn = Mns[idx]
    return Pn, Mn, phi_col*Pn, phi_col*Mn, P0


# ─────────────────────────────────────────────────────────────────────────────
# CURVA M-φ  (Hognestad)
# ─────────────────────────────────────────────────────────────────────────────
def momento_curvatura(b, h, fc, fy, Es, bar_y, Ab_each, N_ax_N=0., n_pts=80):
    ec0 = 0.002; ecu = 0.0035
    bar_d  = h - np.asarray(bar_y, dtype=float)
    As_arr = np.full(len(bar_d), Ab_each)

    def sigc(e_arr):
        e = np.asarray(e_arr, dtype=float).ravel(); s = np.zeros_like(e)
        m1 = (e > 0) & (e <= ec0)
        s[m1] = fc*(2*e[m1]/ec0 - (e[m1]/ec0)**2)
        m2 = e > ec0
        s[m2] = fc*np.clip(1. - 0.15*(e[m2]-ec0)/(ecu-ec0), 0, 1)
        return s

    def equilibrio(c, phi):
        if c <= 0: return -1e12
        yi  = np.linspace(0., c, 80)
        fci = sigc(phi*yi)
        Cc  = mytrapz(fci*b, yi)
        eps_s = phi*(c - bar_d)
        fs    = np.clip(eps_s*Es, -fy, fy)
        fs_net = fs.copy()
        en_c = bar_d < c
        if np.any(en_c):
            fs_net[en_c] -= sigc(phi*np.clip(c-bar_d[en_c], 0, c))
        return Cc + np.sum(fs_net*As_arr) - N_ax_N

    phi_max = ecu / max(0.05*h, 1.)
    phis    = np.linspace(1e-10, phi_max, n_pts)
    Ms, phi_list, c_prev = [], [], 0.45*h

    for phi in phis:
        try:
            c = brentq(lambda cc: equilibrio(cc, phi), 1e-3, h*8,
                       xtol=0.1, maxiter=400)
        except ValueError:
            c = c_prev
        yi  = np.linspace(0., c, 80); fci = sigc(phi*yi)
        Cc  = max(mytrapz(fci*b, yi), 1e-9)
        yc  = mytrapz(fci*b*yi, yi)/Cc
        eps_s = phi*(c - bar_d); fs = np.clip(eps_s*Es, -fy, fy)
        fs_net = fs.copy()
        en_c = bar_d < c
        if np.any(en_c):
            fs_net[en_c] -= sigc(phi*np.clip(c-bar_d[en_c], 0, c))
        M = Cc*(h/2 - yc) + np.sum(fs_net*As_arr*(h/2 - bar_d))
        Ms.append(M); phi_list.append(phi); c_prev = c
        if phi*c > ecu*1.1: break

    return np.array(phi_list), np.array(Ms)/1e6


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTAR EXCEL
# ─────────────────────────────────────────────────────────────────────────────
def to_excel(rows):
    buf = BytesIO()
    df  = pd.DataFrame.from_dict(rows, orient="index", columns=["Valor","Unidades"])
    df.index.name = "Propiedad"
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Propiedades")
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════════════════
# BARRA LATERAL
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    categoria = st.selectbox("Categoría de sección", [
        "🔩 Acero — Perfiles laminados",
        "🏗️ Hormigón / Genérica",
        "🔧 Sección Compuesta (Acero + HA)",
    ])
    st.divider()
    st.markdown("### Dimensiones (mm)")

    # valores por defecto (evitan NameError en paths no ejecutados)
    tipo   = "Perfil I (doble T)"
    d = b  = t_f = t_w = r = t = r_r = r_t = r_out = 200.
    n_r = n = 8
    d_f = b_f = b_w = d_w = b_fb = d_fb = 200.
    d_s = b_s = t_f_s = t_w_s = r_s = b_slab = h_slab = 200.
    E_steel = 200000.; E_conc = 27000.; nu_s = 0.3; nu_c = 0.2
    mesh_size = mesh_size_s = mesh_size_c = 50.
    fc = 25.; fy = 420.; Es_v = 200000.
    n_bars = 6; d_bar = 20.; cover = 40.

    # ── ACERO ─────────────────────────────────────────────────────────────────
    if categoria.startswith("🔩"):
        tipo = st.selectbox("Tipo de perfil", [
            "Perfil I (doble T)","Canal (C / UPN)","Ángulo (L)",
            "Perfil T","RHS — Tubo rectangular","CHS — Tubo circular"])
        if tipo == "Perfil I (doble T)":
            d=st.number_input("d — altura total (mm)",50.,2000.,200.,5.)
            b=st.number_input("b — ancho ala (mm)",20.,1000.,150.,5.)
            t_f=st.number_input("t_f — esp. ala (mm)",1.,100.,10.,.5)
            t_w=st.number_input("t_w — esp. alma (mm)",1.,100.,6.,.5)
            r=st.number_input("r — radio acuerdo (mm)",0.,50.,8.,.5)
            n_r=st.slider("n_r — puntos radio",4,16,8)
        elif tipo == "Canal (C / UPN)":
            d=st.number_input("d",50.,1000.,200.,5.); b=st.number_input("b",20.,500.,75.,5.)
            t_f=st.number_input("t_f",1.,50.,11.,.5); t_w=st.number_input("t_w",1.,50.,6.,.5)
            r=st.number_input("r",0.,30.,10.,.5); n_r=st.slider("n_r",4,16,8)
        elif tipo == "Ángulo (L)":
            b=st.number_input("b — ala horizontal",20.,500.,100.,5.)
            d=st.number_input("d — ala vertical",20.,500.,100.,5.)
            t=st.number_input("t — espesor",1.,50.,10.,.5)
            r_r=st.number_input("r_r — radio raíz",0.,30.,12.,.5)
            r_t=st.number_input("r_t — radio punta",0.,20.,5.,.5)
            n_r=st.slider("n_r",4,16,8)
        elif tipo == "Perfil T":
            d=st.number_input("d — altura total",20.,1000.,150.,5.)
            b=st.number_input("b — ancho ala",20.,500.,100.,5.)
            t_f=st.number_input("t_f",1.,50.,10.,.5); t_w=st.number_input("t_w",1.,50.,6.,.5)
        elif tipo == "RHS — Tubo rectangular":
            d=st.number_input("d — altura",20.,1000.,150.,5.)
            b=st.number_input("b — ancho",20.,1000.,100.,5.)
            t=st.number_input("t — esp. pared",1.,50.,6.,.5)
            r_out=st.number_input("r_out",0.,50.,10.,.5); n_r=st.slider("n_r",4,16,8)
        elif tipo == "CHS — Tubo circular":
            d=st.number_input("d — diámetro ext.",20.,1000.,150.,5.)
            t=st.number_input("t — esp. pared",1.,100.,6.,.5); n=st.slider("n",16,64,32)
        mesh_size=st.number_input("Tamaño de malla (mm²)",0.5,500.,10.,1.)

    # ── HORMIGÓN ──────────────────────────────────────────────────────────────
    elif categoria.startswith("🏗️"):
        tipo = st.selectbox("Tipo de sección",
                            ["Rectangular","Circular","Viga T","Doble T (I-H)"])
        if tipo == "Rectangular":
            b=st.number_input("b — ancho (mm)",50.,5000.,300.,10.)
            d=st.number_input("d — alto  (mm)",50.,5000.,500.,10.)
        elif tipo == "Circular":
            d=st.number_input("d — diámetro (mm)",50.,5000.,400.,10.)
            n=st.slider("n — nodos circunferencia",16,64,32)
        elif tipo == "Viga T":
            b_f=st.number_input("b_f — ancho ala",100.,5000.,600.,10.)
            d_f=st.number_input("d_f — esp. ala",20.,500.,100.,5.)
            b_w=st.number_input("b_w — ancho alma",50.,2000.,200.,5.)
            d_w=st.number_input("d_w — alto alma",100.,5000.,400.,10.)
        elif tipo == "Doble T (I-H)":
            b_f=st.number_input("b_f — ala sup.",100.,5000.,400.,10.)
            d_f=st.number_input("d_f — esp. sup.",20.,500.,80.,5.)
            b_w=st.number_input("b_w — alma",50.,2000.,150.,5.)
            d_w=st.number_input("d_w — alto alma",100.,5000.,400.,10.)
            b_fb=st.number_input("b_fb — ala inf.",100.,5000.,250.,10.)
            d_fb=st.number_input("d_fb — esp. inf.",20.,500.,80.,5.)
        mesh_size=st.number_input("Tamaño de malla (mm²)",10.,50000.,500.,50.)
        st.divider()
        st.markdown("### 🔬 Parámetros HA / Armadura")
        fc=st.number_input("f'c (MPa)",10.,120.,25.,1.)
        fy=st.number_input("fy  (MPa)",200.,600.,420.,10.)
        Es_v=st.number_input("Es  (MPa)",100000.,250000.,200000.,1000.,format="%.0f")
        n_bars=int(st.number_input("N° barras totales",2.,32.,6.,2.))
        d_bar=st.number_input("∅ barra (mm)",6.,50.,20.,2.)
        cover=st.number_input("Recubrimiento (mm)",10.,100.,40.,5.)

    # ── COMPUESTA ─────────────────────────────────────────────────────────────
    elif categoria.startswith("🔧"):
        tipo = "Viga compuesta"
        st.markdown("**Perfil I de Acero**")
        d_s=st.number_input("d",50.,2000.,300.,5.); b_s=st.number_input("b",20.,500.,150.,5.)
        t_f_s=st.number_input("t_f",1.,100.,12.,.5); t_w_s=st.number_input("t_w",1.,100.,8.,.5)
        r_s=st.number_input("r",0.,30.,8.,.5)
        st.markdown("**Losa de Hormigón**")
        b_slab=st.number_input("b_e — ancho ef. (mm)",50.,5000.,1200.,10.)
        h_slab=st.number_input("h_s — espesor (mm)",20.,500.,120.,5.)
        st.markdown("**Materiales**")
        E_steel=st.number_input("E acero (MPa)",100000.,250000.,200000.,format="%.0f")
        E_conc=st.number_input("E hormigón (MPa)",5000.,50000.,27000.,500.)
        nu_s=st.number_input("ν acero",0.1,0.5,0.3,0.01)
        nu_c=st.number_input("ν hormigón",0.1,0.5,0.2,0.01)
        mesh_size_s=st.number_input("Malla acero (mm²)",1.,200.,15.,5.)
        mesh_size_c=st.number_input("Malla hormigón (mm²)",50.,10000.,600.,50.)

    st.divider()
    st.markdown("### ⚡ Cargas aplicadas")
    ca, cb = st.columns(2)
    with ca:
        N_kN  = st.number_input("N (kN) +tracción",-10000.,10000.,0.,10.)
        Mxx_v = st.number_input("Mxx (kN·m)",-5000.,5000.,50.,5.)
        Vx_v  = st.number_input("Vx (kN)",-5000.,5000.,0.,5.)
    with cb:
        Myy_v = st.number_input("Myy (kN·m)",-5000.,5000.,0.,5.)
        Vy_v  = st.number_input("Vy (kN)",-5000.,5000.,50.,5.)
        Mzz_v = st.number_input("Mzz (kN·m — torsión)",-5000.,5000.,0.,5.)

    st.divider()
    run = st.button("🔍 CALCULAR", type="primary", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# BIENVENIDA
# ═════════════════════════════════════════════════════════════════════════════
if not run and "section" not in st.session_state:
    c1,c2,c3 = st.columns(3)
    for col,title,items in [
        (c1,"📐 Secciones disponibles",[
            "Acero: Perfil I, Canal, Ángulo, T, RHS, CHS",
            "Hormigón: Rectangular, Circular, Viga T, Doble T",
            "Compuesta: Viga mixta (Acero + losa HA)"]),
        (c2,"📊 Análisis incluidos",[
            "Propiedades geométricas: A, Ix, Iy, Ze, rx, ry",
            "Alabeo: J, Iw, centro de corte, Av",
            "Propiedades plásticas: Zp, factores de forma",
            "Tensiones: σ_axial, σ_flex, τ, Von Mises"]),
        (c3,"🏗️ Diseño HA",[
            "Diagrama de interacción P-M (ACI 318 / NCh430)",
            "Curva momento-curvatura M-φ (Hognestad)",
            "Ductilidad μφ = φ_u / φ_y",
            "Visualización de armadura y deformaciones"])]:
        with col:
            st.markdown(f"**{title}**")
            for i in items: st.markdown(f"- {i}")
    st.info("👈 Configure los parámetros en la barra lateral y presione **CALCULAR**.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN + ANÁLISIS FEM
# ═════════════════════════════════════════════════════════════════════════════
if run:
    with st.spinner("⏳ Construyendo geometría y generando malla FEM..."):
        try:
            if categoria.startswith("🔩"):
                if   tipo == "Perfil I (doble T)":       geom = i_section(d=d,b=b,t_f=t_f,t_w=t_w,r=r,n_r=n_r)
                elif tipo == "Canal (C / UPN)":           geom = channel_section(d=d,b=b,t_f=t_f,t_w=t_w,r=r,n_r=n_r)
                elif tipo == "Ángulo (L)":                geom = angle_section(b=b,d=d,t=t,r_r=r_r,r_t=r_t,n_r=n_r)
                elif tipo == "Perfil T":                  geom = make_t_steel(d=d,b=b,t_f=t_f,t_w=t_w)
                elif tipo == "RHS — Tubo rectangular":    geom = rectangular_hollow_section(d=d,b=b,t=t,r_out=r_out,n_r=n_r)
                elif tipo == "CHS — Tubo circular":       geom = circular_hollow_section(d=d,t=t,n=n)
                geom.create_mesh(mesh_sizes=[mesh_size])
                section = Section(geometry=geom); is_composite = False

            elif categoria.startswith("🏗️"):
                if   tipo == "Rectangular":   geom = rectangular_section(d=d, b=b)
                elif tipo == "Circular":      geom = circular_section(d=d, n=n)
                elif tipo == "Viga T":        geom = make_concrete_T(b_f=b_f,d_f=d_f,b_w=b_w,d_w=d_w)
                elif tipo == "Doble T (I-H)": geom = make_double_T(b_f=b_f,d_f=d_f,b_w=b_w,d_w=d_w,b_fb=b_fb,d_fb=d_fb)
                geom.create_mesh(mesh_sizes=[mesh_size])
                section = Section(geometry=geom); is_composite = False

            elif categoria.startswith("🔧"):
                sm = Material(name="Acero",elastic_modulus=E_steel,
                              poissons_ratio=nu_s,density=7.85e-6,yield_strength=250,color="steelblue")
                cm = Material(name="Hormigón",elastic_modulus=E_conc,
                              poissons_ratio=nu_c,density=2.4e-6,yield_strength=25,color="lightgray")
                sg = i_section(d=d_s,b=b_s,t_f=t_f_s,t_w=t_w_s,r=r_s,n_r=8).assign_material(sm)
                cg = (rectangular_section(d=h_slab,b=b_slab)
                      .align_center(align_to=sg).align_to(other=sg,on="top")
                      .assign_material(cm))
                geom = sg + cg
                geom.create_mesh(mesh_sizes=[mesh_size_s, mesh_size_c])
                section = Section(geometry=geom); is_composite = True

            # dims HA para P-M y M-φ
            if categoria.startswith("🏗️"):
                if   tipo == "Rectangular":   ha_b,ha_h = b,d
                elif tipo == "Circular":      ha_b,ha_h = d,d
                elif tipo == "Viga T":        ha_b,ha_h = b_w, d_f+d_w
                elif tipo == "Doble T (I-H)": ha_b,ha_h = b_w, d_f+d_w+d_fb
            else:
                ha_b,ha_h = b,d

            st.session_state.update({
                "section":section,"geom":geom,
                "categoria":categoria,"tipo":tipo,"is_composite":is_composite,
                "ha_b":ha_b,"ha_h":ha_h,"ha_fc":fc,"ha_fy":fy,"ha_Es":Es_v,
                "ha_n_bars":n_bars,"ha_d_bar":d_bar,"ha_cover":cover,
                "E_ref":E_steel if categoria.startswith("🔧") else 1.,
                "N_kN":N_kN,"Mxx_v":Mxx_v,"Myy_v":Myy_v,
                "Vx_v":Vx_v,"Vy_v":Vy_v,"Mzz_v":Mzz_v,
            })
        except Exception as e:
            st.error(f"❌ Error al construir la geometría: {e}")
            st.stop()

# recuperar estado
section      = st.session_state["section"]
geom         = st.session_state["geom"]
cat_s        = st.session_state["categoria"]
tip_s        = st.session_state["tipo"]
is_composite = st.session_state["is_composite"]
E_ref        = st.session_state.get("E_ref", 1.)
N_kN   = st.session_state.get("N_kN",   N_kN)
Mxx_v  = st.session_state.get("Mxx_v",  Mxx_v)
Myy_v  = st.session_state.get("Myy_v",  Myy_v)
Vx_v   = st.session_state.get("Vx_v",   Vx_v)
Vy_v   = st.session_state.get("Vy_v",   Vy_v)
Mzz_v  = st.session_state.get("Mzz_v",  Mzz_v)

# análisis
with st.spinner("📐 Propiedades geométricas..."):    section.calculate_geometric_properties()
warp_ok = False
with st.spinner("🌀 Propiedades de alabeo..."):
    try: section.calculate_warping_properties(); warp_ok = True
    except: pass
plas_ok = False
with st.spinner("💪 Propiedades plásticas..."):
    try: section.calculate_plastic_properties(); plas_ok = True
    except: pass

st.success(f"✅ Análisis completado — {section.num_nodes} nodos · {len(section.elements)} elementos FEM")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_labels = ["📊 Geometría","📋 Propiedades","🌡️ Tensiones"]
if cat_s.startswith("🏗️"): tab_labels += ["📈 Diagrama P-M","📉 Curva M-φ"]
tabs = st.tabs(tab_labels)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  ·  GEOMETRÍA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Geometría y Malla de Elementos Finitos")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Sección transversal**")
        fig, ax = plt.subplots(figsize=(5.5,5.5))
        try: geom.plot_geometry(ax=ax)
        except: ax.text(0.5,0.5,"No disponible",ha="center",va="center",transform=ax.transAxes)
        ax.set_aspect("equal"); ax.set_title("Geometría",fontsize=10)
        plt.tight_layout(); show_fig(fig)
    with g2:
        st.markdown("**Malla FEM**")
        fig, ax = plt.subplots(figsize=(5.5,5.5))
        try: section.plot_mesh(ax=ax)
        except: ax.text(0.5,0.5,"No disponible",ha="center",va="center",transform=ax.transAxes)
        ax.set_aspect("equal")
        ax.set_title(f"Malla FEM ({section.num_nodes} nodos · {len(section.elements)} elem.)",fontsize=10)
        plt.tight_layout(); show_fig(fig)

    st.markdown("**Centroides y ejes principales**")
    try:
        fig, ax = plt.subplots(figsize=(7,6))
        section.plot_centroids(ax=ax); ax.set_aspect("equal"); plt.tight_layout()
        show_fig(fig,"● centroide elástico · ✦ centro de corte · ■ centroide plástico")
    except Exception as e:
        st.caption(f"Centroides no disponibles: {e}")

    # Armadura para sección rectangular HA
    if cat_s.startswith("🏗️") and tip_s == "Rectangular":
        st.markdown("**Disposición de armadura**")
        b_rc=float(st.session_state["ha_b"]); h_rc=float(st.session_state["ha_h"])
        nb=int(st.session_state["ha_n_bars"]); db=float(st.session_state["ha_d_bar"])
        cov=float(st.session_state["ha_cover"])
        bar_y_draw, bar_x_draw = distribuir_barras(b_rc,h_rc,nb,cov,db)
        try: cx_d,cy_d = section.get_c()
        except: cx_d=cy_d=None
        fig, ax = plt.subplots(figsize=(5,6))
        dibujar_seccion_ha(ax,b_rc,h_rc,bar_y_draw,bar_x_draw,db,cov,
                           titulo=f"Sección {b_rc:.0f}×{h_rc:.0f} mm — {nb}∅{db:.0f} mm",
                           cx=cx_d,cy=cy_d)
        plt.tight_layout()
        show_fig(fig, f"As total = {nb*Ab_circ(db):.0f} mm²  ·  "
                      f"ρ = {nb*Ab_circ(db)/(b_rc*h_rc)*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  ·  PROPIEDADES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Propiedades de la Sección")
    pg, pw, pp = st.columns(3)
    all_results = {}

    # ── Geométricas ────────────────────────────────────────────────────────────
    with pg:
        st.markdown("#### 📐 Geométricas")
        try:
            A  = section.get_area() if not is_composite else section.get_ea()/E_ref
            cx, cy = section.get_c()
            pcard("Área A", A, "mm²"); pcard("cx", cx, "mm"); pcard("cy", cy, "mm")
            all_results.update({"A [mm²]":(fmt_si(A),"mm²"),
                                 "cx [mm]":(fmt_si(cx),"mm"),"cy [mm]":(fmt_si(cy),"mm")})

            if not is_composite:
                ixx,iyy,ixy = section.get_ic()
                pcard("Ixx centroidal",ixx,"mm⁴"); pcard("Iyy centroidal",iyy,"mm⁴")
                pcard("Ixy centroidal",ixy,"mm⁴")
                all_results.update({"Ixx [mm⁴]":(fmt_si(ixx),"mm⁴"),
                                     "Iyy [mm⁴]":(fmt_si(iyy),"mm⁴"),
                                     "Ixy [mm⁴]":(fmt_si(ixy),"mm⁴")})
                try:
                    ip = section.get_ip()          # v3.x → (i11, i22)
                    i11,i22 = float(ip[0]),float(ip[1])
                    phi_ang = section.get_phi()
                    pcard("I₁₁ eje principal",i11,"mm⁴"); pcard("I₂₂ eje principal",i22,"mm⁴")
                    pcard("φ eje principal",phi_ang,"°")
                    all_results.update({"I₁₁ [mm⁴]":(fmt_si(i11),"mm⁴"),
                                         "I₂₂ [mm⁴]":(fmt_si(i22),"mm⁴"),
                                         "φ_ppal [°]":(fmt_si(phi_ang),"°")})
                except Exception: pass
                try:
                    zp,zm,zyp,zym = section.get_z()
                    pcard("Zxx+ módulo sup.",zp,"mm³"); pcard("Zxx− módulo inf.",zm,"mm³")
                    pcard("Zyy+",zyp,"mm³"); pcard("Zyy−",zym,"mm³")
                    all_results.update({"Zxx+ [mm³]":(fmt_si(zp),"mm³"),
                                         "Zxx− [mm³]":(fmt_si(zm),"mm³"),
                                         "Zyy+ [mm³]":(fmt_si(zyp),"mm³"),
                                         "Zyy− [mm³]":(fmt_si(zym),"mm³")})
                except Exception: pass
                try:
                    rx,ry = section.get_rc()
                    pcard("rx radio de giro",rx,"mm"); pcard("ry radio de giro",ry,"mm")
                    all_results.update({"rx [mm]":(fmt_si(rx),"mm"),"ry [mm]":(fmt_si(ry),"mm")})
                except Exception: pass
                try:
                    Qx,Qy = section.get_q()
                    pcard("Qx",Qx,"mm³"); pcard("Qy",Qy,"mm³")
                    all_results.update({"Qx [mm³]":(fmt_si(Qx),"mm³"),"Qy [mm³]":(fmt_si(Qy),"mm³")})
                except Exception: pass
            else:
                eixx,eiyy,eixy = section.get_eic(e_ref=E_ref)
                pcard(f"E·Ixx (E_ref={E_ref:.0f})",eixx,"N·mm²")
                pcard(f"E·Iyy (E_ref={E_ref:.0f})",eiyy,"N·mm²")
                pcard(f"Ixx transf.",eixx/E_ref,"mm⁴"); pcard(f"Iyy transf.",eiyy/E_ref,"mm⁴")
                try:
                    eac = section.get_ea()
                    pcard("E·A rigidez axial",eac,"N")
                    all_results["E·A [N]"] = (fmt_si(eac),"N")
                except Exception: pass
                all_results.update({"E·Ixx [N·mm²]":(fmt_si(eixx),"N·mm²"),
                                     "E·Iyy [N·mm²]":(fmt_si(eiyy),"N·mm²"),
                                     "Ixx transf. [mm⁴]":(fmt_si(eixx/E_ref),"mm⁴"),
                                     "Iyy transf. [mm⁴]":(fmt_si(eiyy/E_ref),"mm⁴")})
        except Exception as e:
            st.warning(f"Propiedades geométricas: {e}")

    # ── Alabeo ────────────────────────────────────────────────────────────────
    with pw:
        st.markdown("#### 🌀 Alabeo / Warping")
        if warp_ok:
            for name,fn,unit,key in [
                ("J — constante de torsión", lambda: section.get_j(),      "mm⁴",  "J [mm⁴]"),
                ("Iw — constante de alabeo",  lambda: section.get_gamma(),  "mm⁶",  "Iw [mm⁶]"),
            ]:
                try:
                    v = fn(); pcard(name,v,unit); all_results[key] = (fmt_si(v),unit)
                except Exception: pass
            try:
                xs,ys = section.get_sc()
                pcard("xs centro de corte",xs,"mm"); pcard("ys centro de corte",ys,"mm")
                all_results.update({"xs [mm]":(fmt_si(xs),"mm"),"ys [mm]":(fmt_si(ys),"mm")})
            except Exception: pass
            try:
                Asx,Asy = section.get_as()
                pcard("Asx área de corte x",Asx,"mm²"); pcard("Asy área de corte y",Asy,"mm²")
                all_results.update({"Asx [mm²]":(fmt_si(Asx),"mm²"),"Asy [mm²]":(fmt_si(Asy),"mm²")})
            except Exception: pass
            try:
                x1se,y2se = section.get_sc_p()
                pcard("x₁_se c. corte ppal.",x1se,"mm"); pcard("y₂_se c. corte ppal.",y2se,"mm")
            except Exception: pass
        else:
            st.markdown('<div class="warn-box">⚠️ Propiedades de alabeo no calculadas</div>',
                        unsafe_allow_html=True)

    # ── Plásticas ─────────────────────────────────────────────────────────────
    with pp:
        st.markdown("#### 💪 Plásticas")
        if plas_ok:
            if not is_composite:
                Sxx = Syy = None
                try:
                    Sxx,Syy = section.get_sp()
                    pcard("Zpx módulo plástico x",Sxx,"mm³"); pcard("Zpy módulo plástico y",Syy,"mm³")
                    all_results.update({"Zpx [mm³]":(fmt_si(Sxx),"mm³"),"Zpy [mm³]":(fmt_si(Syy),"mm³")})
                except Exception: pass
                # factores de forma = Zp / Ze
                try:
                    zp_e,zm_e,zyp_e,zym_e = section.get_z()
                    if Sxx and abs(zp_e)>0:
                        sfxp = Sxx/zp_e; sfxm = Sxx/abs(zm_e) if abs(zm_e)>0 else float("nan")
                        pcard("SF_xx+ factor de forma",sfxp); pcard("SF_xx− factor de forma",sfxm)
                        all_results.update({"SF_xx+":(fmt_si(sfxp),""),"SF_xx−":(fmt_si(sfxm),"")})
                    if Syy and abs(zyp_e)>0:
                        sfyp = Syy/zyp_e; sfym = Syy/abs(zym_e) if abs(zym_e)>0 else float("nan")
                        pcard("SF_yy+ factor de forma",sfyp); pcard("SF_yy− factor de forma",sfym)
                        all_results.update({"SF_yy+":(fmt_si(sfyp),""),"SF_yy−":(fmt_si(sfym),"")})
                except Exception: pass
                try:
                    xpc,ypc = section.get_pc()
                    pcard("x centroide plástico",xpc,"mm"); pcard("y centroide plástico",ypc,"mm")
                    all_results.update({"xpc [mm]":(fmt_si(xpc),"mm"),"ypc [mm]":(fmt_si(ypc),"mm")})
                except Exception: pass
            else:
                try:
                    mp11,mp22 = section.get_mp()
                    pcard("Mp_xx momento plástico",mp11,"N·mm"); pcard("Mp_yy momento plástico",mp22,"N·mm")
                    all_results.update({"Mp_xx [N·mm]":(fmt_si(mp11),"N·mm"),"Mp_yy [N·mm]":(fmt_si(mp22),"N·mm")})
                except Exception:
                    st.info("Momento plástico no disponible para esta configuración.")
        else:
            st.markdown('<div class="warn-box">⚠️ Propiedades plásticas no calculadas</div>',
                        unsafe_allow_html=True)

    # ── Tabla resumen + exportar ──────────────────────────────────────────────
    if all_results:
        st.markdown("---")
        st.markdown("#### 📄 Tabla resumen de propiedades")
        df_res = pd.DataFrame.from_dict(all_results, orient="index",
                                         columns=["Valor","Unidades"])
        df_res.index.name = "Propiedad"
        st.dataframe(df_res, use_container_width=True)
        st.download_button(
            label="⬇️ Exportar a Excel",
            data=to_excel(all_results),
            file_name="propiedades_seccion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  ·  TENSIONES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Visualización de Tensiones")
    any_load = any(abs(v)>0 for v in [N_kN,Mxx_v,Myy_v,Vx_v,Vy_v,Mzz_v])
    if not any_load:
        st.info("👈 Ingresa al menos una carga en la barra lateral y recalcula.")
    else:
        with st.spinner("Calculando tensiones..."):
            try:
                stress_post = section.calculate_stress(
                    N=N_kN*1e3, Mxx=Mxx_v*1e6, Myy=Myy_v*1e6,
                    Vx=Vx_v*1e3, Vy=Vy_v*1e3, Mzz=Mzz_v*1e6)
                stress_ok = True
            except Exception as e:
                st.error(f"Error calculando tensiones: {e}"); stress_ok = False

        if stress_ok:
            STRESS_MAP = {
                "σ normal combinada (N + Mxx + Myy)": "sig_zz",
                "σ axial (N)":                        "sig_zz_n",
                "σ flexión Mxx":                      "sig_zz_mxx",
                "σ flexión Myy":                      "sig_zz_myy",
                "τ cortante Vy":                      "sig_zxy_vy",
                "τ cortante Vx":                      "sig_zxy_vx",
                "τ torsión Mzz":                      "sig_zxy_mzz",
                "τ combinado total":                  "sig_zxy",
                "Tensión de Von Mises":               "sig_vm",
            }
            sel = st.selectbox("Componente de tensión", list(STRESS_MAP))
            key = STRESS_MAP[sel]
            sv1, sv2 = st.columns([3,1])
            with sv1:
                try:
                    fig, ax = plt.subplots(figsize=(6.5,6.5))
                    stress_post.plot_stress(stress=key, ax=ax, colorbar=True)
                    ax.set_aspect("equal"); ax.set_title(sel, fontsize=9)
                    plt.tight_layout(); show_fig(fig)
                except Exception as e:
                    st.warning(f"No se puede graficar: {e}")
                if "zxy" in key:
                    try:
                        fig2, ax2 = plt.subplots(figsize=(6.5,6.5))
                        stress_post.plot_stress_vector(stress=key, ax=ax2)
                        ax2.set_aspect("equal"); ax2.set_title(f"Vectores — {sel}",fontsize=9)
                        plt.tight_layout(); show_fig(fig2)
                    except Exception: pass
            with sv2:
                st.markdown("**Cargas ingresadas**")
                st.table(pd.DataFrame({
                    "Carga": ["N","Mxx","Myy","Vx","Vy","Mzz"],
                    "Valor": [N_kN,Mxx_v,Myy_v,Vx_v,Vy_v,Mzz_v],
                    "Unidad":["kN","kN·m","kN·m","kN","kN","kN·m"],
                }))
                try:
                    sig = stress_post.get_stress()
                    vals = sig[0][key]
                    st.metric("σ_máx", f"{np.max(vals):.4g} MPa")
                    st.metric("σ_mín", f"{np.min(vals):.4g} MPa")
                except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# TABS 4 y 5  ·  SÓLO HORMIGÓN
# ══════════════════════════════════════════════════════════════════════════════
if cat_s.startswith("🏗️"):
    b_rc  = float(st.session_state["ha_b"])
    h_rc  = float(st.session_state["ha_h"])
    fc_v  = float(st.session_state["ha_fc"])
    fy_v  = float(st.session_state["ha_fy"])
    Es_v2 = float(st.session_state["ha_Es"])
    nb    = int(st.session_state["ha_n_bars"])
    db    = float(st.session_state["ha_d_bar"])
    cov   = float(st.session_state["ha_cover"])
    bar_y_rc, bar_x_rc = distribuir_barras(b_rc, h_rc, nb, cov, db)
    Ab_bar = Ab_circ(db)
    rho_g  = nb * Ab_bar / (b_rc * h_rc) * 100

    # ─── TAB 4  ·  DIAGRAMA P-M ───────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Diagrama de Interacción P-M")
        st.caption(f"b = {b_rc:.0f} mm · h = {h_rc:.0f} mm · "
                   f"{nb}∅{db:.0f} mm · f'c = {fc_v} MPa · fy = {fy_v} MPa · ρ = {rho_g:.2f}%")

        pm1, pm2 = st.columns([3,1])
        with pm2:
            phi_c = st.slider("Factor φ", 0.50, 1.00, 0.65, 0.01,
                               help="φ = 0.65 estribos | φ = 0.75 espiral | φ = 1.0 nominal")
            fig_s, ax_s = plt.subplots(figsize=(3.5,4.5))
            dibujar_seccion_ha(ax_s, b_rc, h_rc, bar_y_rc, bar_x_rc, db, cov,
                               titulo=f"{nb}∅{db:.0f} mm")
            plt.tight_layout(); show_fig(fig_s)
            st.caption(f"As = {nb*Ab_bar:.0f} mm²  ·  ρ = {rho_g:.2f}%")

        with pm1:
            with st.spinner("Calculando..."):
                try:
                    Pn,Mn,phiPn,phiMn,P0 = rc_pm_diagram(
                        b=b_rc, h=h_rc, fc=fc_v, fy=fy_v, Es=Es_v2,
                        bar_y=bar_y_rc, Ab_each=Ab_bar, phi_col=phi_c)

                    fig, ax = plt.subplots(figsize=(7.5,8))
                    ax.fill_betweenx(phiPn, phiMn, alpha=0.1, color="#1565C0")
                    ax.plot(phiMn, phiPn, color="#1565C0", lw=2.5,
                            label=f"φRn  (φ = {phi_c})")
                    ax.plot(Mn, Pn, "--", color="#90A4AE", lw=1.3, alpha=0.9,
                            label="Resistencia nominal Rn")
                    ax.plot(abs(Mxx_v), N_kN, "r*", ms=16, zorder=6,
                            label=f"Demanda  P = {N_kN:.0f} kN, M = {abs(Mxx_v):.1f} kN·m")
                    ax.axhline(0, color="gray", lw=0.7, ls=":")
                    ax.axvline(0, color="gray", lw=0.7, ls=":")
                    idx_peak = int(np.argmax(Mn))
                    ax.plot(Mn[idx_peak], Pn[idx_peak], "^", color="#F57C00",
                            ms=11, zorder=5,
                            label=f"Balance  P ≈ {Pn[idx_peak]:.0f} kN")
                    ax.set_xlabel("Momento M (kN·m)", fontsize=11)
                    ax.set_ylabel("Fuerza Axial P (kN)", fontsize=11)
                    ax.set_title(f"Diagrama de Interacción P-M\n"
                                 f"b = {b_rc:.0f} mm · h = {h_rc:.0f} mm · "
                                 f"f'c = {fc_v} MPa · fy = {fy_v} MPa", fontsize=10)
                    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
                    plt.tight_layout(); show_fig(fig)

                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("P₀ compresión pura",  f"{P0:.0f} kN")
                    m2.metric("φP₀ (reducida)",      f"{phi_c*P0:.0f} kN")
                    m3.metric("Mn máximo",            f"{max(Mn):.1f} kN·m")
                    m4.metric("Pt tracción pura",     f"{min(Pn):.1f} kN")

                    # verificación demanda
                    Mpd, Ppd = abs(Mxx_v), N_kN
                    inside = False
                    if len(phiMn) > 5:
                        try:
                            idx_s = np.argsort(phiPn)
                            fi = interp1d(phiPn[idx_s], phiMn[idx_s],
                                          bounds_error=False, fill_value=0.)
                            inside = (Mpd <= float(fi(Ppd))) and (phi_c*P0 >= Ppd >= min(phiPn))
                        except Exception: pass
                    if abs(Mxx_v)+abs(N_kN) > 0:
                        if inside:
                            st.markdown('<div class="ok-box">✅ Punto de demanda DENTRO de la envolvente φRn</div>',
                                        unsafe_allow_html=True)
                        else:
                            st.error("❌ Punto de demanda FUERA de la envolvente φRn")

                    # tabla puntos notables
                    st.markdown("---")
                    st.markdown("#### Puntos notables del diagrama")
                    df_pm = pd.DataFrame({
                        "Punto": ["Compresión pura (P₀)","Balance (Mn_max)","Tracción pura"],
                        "Pn (kN)": [P0, float(Pn[idx_peak]), float(min(Pn))],
                        "Mn (kN·m)": [0., float(Mn[idx_peak]), 0.],
                        "φPn (kN)": [phi_c*P0, phi_c*float(Pn[idx_peak]), phi_c*float(min(Pn))],
                        "φMn (kN·m)": [0., phi_c*float(Mn[idx_peak]), 0.],
                    })
                    st.dataframe(df_pm.style.format({
                        "Pn (kN)":"{:.1f}","Mn (kN·m)":"{:.2f}",
                        "φPn (kN)":"{:.1f}","φMn (kN·m)":"{:.2f}"}),
                        use_container_width=True)

                except Exception as e:
                    st.error(f"Error en diagrama de interacción: {e}")

    # ─── TAB 5  ·  CURVA M-φ ──────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Curva Momento-Curvatura  M-φ")
        st.caption(f"b = {b_rc:.0f} mm · h = {h_rc:.0f} mm · "
                   f"{nb}∅{db:.0f} mm · f'c = {fc_v} MPa · fy = {fy_v} MPa")

        mf1, mf2 = st.columns([3,1])
        with mf2:
            N_mf    = st.number_input("N axial (kN, + compresión)",
                                       -5000., 50000., float(max(0.,-N_kN)), 10.,
                                       key="N_mf_input")
            n_pts_v = st.slider("Pts. integración", 40, 150, 80, 10)
            fig_s2, ax_s2 = plt.subplots(figsize=(3.5,4.5))
            dibujar_seccion_ha(ax_s2, b_rc, h_rc, bar_y_rc, bar_x_rc, db, cov,
                               titulo=f"{nb}∅{db:.0f} mm")
            plt.tight_layout(); show_fig(fig_s2)
            st.caption(f"As = {nb*Ab_bar:.0f} mm²  ·  ρ = {rho_g:.2f}%")

        with mf1:
            with st.spinner("Calculando curva M-φ..."):
                try:
                    phi_arr, M_arr = momento_curvatura(
                        b=b_rc, h=h_rc, fc=fc_v, fy=fy_v, Es=Es_v2,
                        bar_y=bar_y_rc, Ab_each=Ab_bar,
                        N_ax_N=N_mf*1e3, n_pts=n_pts_v)

                    M_max = float(np.max(M_arr))
                    i_max = int(np.argmax(M_arr)); phi_u = float(phi_arr[i_max])
                    ey    = fy_v/Es_v2
                    d_eff = h_rc - (cov + db/2)
                    c_y   = ey*d_eff/(0.003+ey)
                    phi_y = ey/max(d_eff-c_y, 1e-9)
                    M_y   = float(np.interp(phi_y, phi_arr, M_arr))
                    mu_phi = phi_u/phi_y if phi_y>0 else float("nan")
                    EI_init = M_arr[2]*1e6/phi_arr[2]/1e9 if len(phi_arr)>3 and phi_arr[2]>0 else float("nan")

                    # ── Gráfico principal ──────────────────────────────────────
                    fig, ax = plt.subplots(figsize=(8,5))
                    ax.plot(phi_arr*1e6, M_arr, color="#1565C0", lw=2.5, label="M-φ (Hognestad)")
                    ax.fill_between(phi_arr*1e6, M_arr, alpha=0.08, color="#1565C0")
                    # tangente inicial
                    if len(phi_arr) > 4:
                        pt = min(4, len(phi_arr)-1)
                        ax.plot([0, phi_arr[pt]*1e6],[0, M_arr[pt]],":",
                                color="#78909C",lw=1.3,label="Tangente inicial (EI)")
                    ax.axvline(phi_y*1e6, color="#F57C00", ls="--", lw=1.5,
                               label=f"Fluencia  φ_y = {phi_y*1e6:.4f}")
                    ax.plot(phi_y*1e6, M_y, "o", color="#F57C00", ms=11, zorder=5)
                    ax.plot(phi_u*1e6, M_max, "r*", ms=16, zorder=5,
                            label=f"φ_u = {phi_u*1e6:.4f} | Mu = {M_max:.1f} kN·m")
                    if abs(Mxx_v) > 0:
                        ax.axhline(abs(Mxx_v), color="purple", ls="-.", lw=1.2, alpha=0.7,
                                   label=f"Mdemanda = {abs(Mxx_v):.1f} kN·m")
                    ax.set_xlabel("Curvatura φ  (×10⁻⁶ mm⁻¹)", fontsize=11)
                    ax.set_ylabel("Momento M (kN·m)", fontsize=11)
                    ax.set_title(f"Curva Momento-Curvatura\n"
                                 f"N = {N_mf:.0f} kN · f'c = {fc_v} MPa · fy = {fy_v} MPa",
                                 fontsize=10)
                    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25)
                    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
                    plt.tight_layout(); show_fig(fig)

                    mc1,mc2,mc3,mc4 = st.columns(4)
                    mc1.metric("Mu (momento máx.)",         f"{M_max:.2f} kN·m")
                    mc2.metric("φ_y (curvatura fluencia)",  f"{phi_y*1e6:.4f} ×10⁻⁶/mm")
                    mc3.metric("φ_u (curvatura última)",    f"{phi_u*1e6:.4f} ×10⁻⁶/mm")
                    mc4.metric("μφ = φ_u / φ_y",           f"{mu_phi:.2f}")

                    # ── Tabla resumen + deformaciones ──────────────────────────
                    st.markdown("---")
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        st.markdown("#### Parámetros de ductilidad")
                        st.table(pd.DataFrame({
                            "Parámetro":[
                                "Momento fluencia My","Momento último Mu",
                                "Curvatura fluencia φ_y","Curvatura última φ_u",
                                "Ductilidad μφ","Rigidez EI inicial"],
                            "Valor":[
                                f"{M_y:.2f} kN·m", f"{M_max:.2f} kN·m",
                                f"{phi_y*1e6:.4f} ×10⁻⁶/mm", f"{phi_u*1e6:.4f} ×10⁻⁶/mm",
                                f"{mu_phi:.2f}",
                                f"{EI_init:.1f} kN·m²" if not np.isnan(EI_init) else "—"],
                        }))

                    with tc2:
                        st.markdown("#### Distribución de deformaciones")
                        fig3, axes3 = plt.subplots(1, 2, figsize=(6,5), sharey=True)
                        for axi, phi_pt, M_pt, lbl, col in [
                            (axes3[0], phi_y, M_y, "Fluencia", "#F57C00"),
                            (axes3[1], phi_u, M_max, "Rotura", "#D32F2F"),
                        ]:
                            if phi_pt == phi_y:
                                c_pt = ey*d_eff/(0.003+ey)
                            else:
                                c_pt = 0.003/max(phi_pt,1e-12)
                            c_pt = min(max(c_pt, 0.01*h_rc), h_rc)
                            eps_t = phi_pt*c_pt; eps_b = -phi_pt*(h_rc-c_pt)
                            y_p = np.array([0.,h_rc]); e_p = np.array([eps_b,eps_t])
                            axi.plot(e_p*1000, y_p, color=col, lw=2.5)
                            axi.fill_betweenx(y_p, e_p*1000, alpha=0.15, color=col)
                            axi.axvline(0, color="gray", lw=0.7, ls=":")
                            for yi_bar in bar_y_rc:
                                eps_bar = phi_pt*(c_pt-(h_rc-yi_bar))
                                axi.plot(eps_bar*1000, yi_bar, "o",
                                         color="#1565C0", ms=8, zorder=5)
                            axi.set_title(f"{lbl}\nM = {M_pt:.1f} kN·m", fontsize=9)
                            axi.set_xlabel("ε (×10⁻³)", fontsize=8)
                            axi.grid(True, alpha=0.25)
                            axi.set_ylim(-h_rc*0.05, h_rc*1.05)
                        axes3[0].set_ylabel("Altura y (mm)", fontsize=8)
                        plt.tight_layout()
                        show_fig(fig3, "● posición de barras de acero en la sección")

                    # Exportar curva M-φ a CSV
                    df_mphi = pd.DataFrame({
                        "phi_x1e6 [1/mm]": phi_arr*1e6,
                        "M [kN·m]": M_arr,
                    })
                    st.download_button(
                        "⬇️ Exportar curva M-φ a CSV",
                        data=df_mphi.to_csv(index=False),
                        file_name="curva_M_phi.csv", mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Error en curva M-φ: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PIE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📐 **Análisis de Secciones Transversales** · Motor: `sectionproperties` v3.x (FEM) "
    "· Diagrama P-M: Bloque de Whitney (ACI 318 / NCh430) "
    "· Curva M-φ: Modelo de Hognestad (parábola-rectángulo) "
    "· ⚠️ El usuario es responsable de verificar y validar todos los resultados."
)
