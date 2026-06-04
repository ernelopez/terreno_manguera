import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop


# --------------------------------------------------
# Dijkstra sobre una grilla 8-conexa
# --------------------------------------------------

def dijkstra(obstacle, source):
    ny, nx = obstacle.shape

    dist = np.full((ny, nx), np.inf)

    sy, sx = source

    if obstacle[sy, sx]:
        return dist

    pq = [(0, sy, sx)]
    dist[sy, sx] = 0

    vecinos = [
        (-1, 0, 1),
        (1, 0, 1),
        (0, -1, 1),
        (0, 1, 1),
        (-1, -1, np.sqrt(2)),
        (-1, 1, np.sqrt(2)),
        (1, -1, np.sqrt(2)),
        (1, 1, np.sqrt(2)),
    ]

    while pq:

        d, y, x = heappop(pq)

        if d > dist[y, x]:
            continue

        for dy, dx, costo in vecinos:

            y2 = y + dy
            x2 = x + dx

            if not (0 <= y2 < ny and 0 <= x2 < nx):
                continue

            if obstacle[y2, x2]:
                continue

            nd = d + costo

            if nd < dist[y2, x2]:
                dist[y2, x2] = nd
                heappush(pq, (nd, y2, x2))

    return dist


# --------------------------------------------------
# Interfaz
# --------------------------------------------------

st.set_page_config(layout="wide")

st.title("Mapa de distancias evitando un rectángulo interior")

# --------------------------------------------------
# Terreno
# --------------------------------------------------

st.sidebar.header("Terreno")

W = st.sidebar.number_input(
    "Ancho del terreno",
    min_value=10.0,
    max_value=500.0,
    value=100.0,
)

H = st.sidebar.number_input(
    "Alto del terreno",
    min_value=10.0,
    max_value=500.0,
    value=60.0,
)

# --------------------------------------------------
# Obstáculo
# --------------------------------------------------

st.sidebar.header("Rectángulo interior")

rw = st.sidebar.number_input(
    "Ancho del rectángulo",
    min_value=1.0,
    max_value=W / 2,
    value=min(20.0, W / 2),
)

rh = st.sidebar.number_input(
    "Alto del rectángulo",
    min_value=1.0,
    max_value=H / 2,
    value=min(15.0, H / 2),
)

rx = st.sidebar.slider(
    "Posición X",
    0.0,
    float(W - rw),
    float((W - rw) / 2),
)

ry = st.sidebar.slider(
    "Posición Y",
    0.0,
    float(H - rh),
    float((H - rh) / 2),
)

# Garantiza que no toque los bordes

margen = 0.01

rx = min(max(rx, margen), W - rw - margen)
ry = min(max(ry, margen), H - rh - margen)

# --------------------------------------------------
# Resolución
# --------------------------------------------------

st.sidebar.header("Resolución")

N = st.sidebar.slider(
    "Resolución horizontal",
    50,
    200,
    150,
)

M = max(50, int(round(H / W * N)))

# --------------------------------------------------
# Fuente
# --------------------------------------------------

st.sidebar.header("Punto fuente")

lado = st.sidebar.selectbox(
    "Borde",
    ["Inferior", "Superior", "Izquierdo", "Derecho"]
)

t = st.sidebar.slider(
    "Posición sobre el borde (%)",
    0,
    100,
    50
)

if lado == "Inferior":
    x0 = W * t / 100
    y0 = 0

elif lado == "Superior":
    x0 = W * t / 100
    y0 = H

elif lado == "Izquierdo":
    x0 = 0
    y0 = H * t / 100

else:
    x0 = W
    y0 = H * t / 100

# --------------------------------------------------
# Grilla
# --------------------------------------------------

x = np.linspace(0, W, N)
y = np.linspace(0, H, M)

X, Y = np.meshgrid(x, y)

obstacle = (
    (X >= rx)
    & (X <= rx + rw)
    & (Y >= ry)
    & (Y <= ry + rh)
)

sx = np.argmin(np.abs(x - x0))
sy = np.argmin(np.abs(y - y0))

# --------------------------------------------------
# Cálculo automático (sin botón)
# --------------------------------------------------

dist = dijkstra(obstacle, (sy, sx))

dx = W / (N - 1)
dy = H / (M - 1)

dist *= (dx + dy) / 2

dist_plot = dist.copy()
dist_plot[np.isinf(dist_plot)] = np.nan

# --------------------------------------------------
# Gráfico
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

im = ax.imshow(
    dist_plot,
    origin="lower",
    extent=[0, W, 0, H],
    cmap="turbo",
)

rect = plt.Rectangle(
    (rx, ry),
    rw,
    rh,
    facecolor="black",
)

ax.add_patch(rect)

ax.plot(
    x0,
    y0,
    "wo",
    markersize=8,
)

ax.set_xlim(0, W)
ax.set_ylim(0, H)

ax.set_aspect("equal")

ax.set_title("Distancia mínima evitando el obstáculo")

plt.colorbar(
    im,
    ax=ax,
    label="Distancia"
)

st.pyplot(fig)
