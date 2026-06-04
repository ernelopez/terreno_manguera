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
    pq = []

    sy, sx = source

    if obstacle[sy, sx]:
        return dist

    dist[sy, sx] = 0
    heappush(pq, (0, sy, sx))

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

            ny2 = y + dy
            nx2 = x + dx

            if not (0 <= ny2 < ny and 0 <= nx2 < nx):
                continue

            if obstacle[ny2, nx2]:
                continue

            nd = d + costo

            if nd < dist[ny2, nx2]:
                dist[ny2, nx2] = nd
                heappush(pq, (nd, ny2, nx2))

    return dist


# --------------------------------------------------
# Interfaz
# --------------------------------------------------

st.title("Distancias evitando un rectángulo interior")

st.sidebar.header("Terreno")

W = st.sidebar.number_input(
    "Ancho del terreno",
    min_value=1.0,
    value=100.0
)

H = st.sidebar.number_input(
    "Alto del terreno",
    min_value=1.0,
    value=60.0
)

st.sidebar.header("Rectángulo interior")

rx = st.sidebar.number_input(
    "x inferior izquierda",
    value=35.0
)

ry = st.sidebar.number_input(
    "y inferior izquierda",
    value=20.0
)

rw = st.sidebar.number_input(
    "ancho",
    min_value=1.0,
    value=20.0
)

rh = st.sidebar.number_input(
    "alto",
    min_value=1.0,
    value=15.0
)

st.sidebar.header("Resolución")

N = st.sidebar.slider(
    "Cantidad de celdas horizontales",
    50,
    300,
    150
)

M = int(round(H/W*N))

# --------------------------------------------------
# Punto fuente sobre el borde
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
    x0 = W*t/100
    y0 = 0

elif lado == "Superior":
    x0 = W*t/100
    y0 = H

elif lado == "Izquierdo":
    x0 = 0
    y0 = H*t/100

else:
    x0 = W
    y0 = H*t/100


# --------------------------------------------------
# Construcción de la grilla
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
# Cálculo
# --------------------------------------------------

if st.button("Calcular"):

    dist = dijkstra(obstacle, (sy, sx))

    dx = W/(N-1)
    dy = H/(M-1)

    escala = (dx + dy)/2

    dist = dist * escala

    dist_plot = dist.copy()
    dist_plot[np.isinf(dist_plot)] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        dist_plot,
        origin="lower",
        extent=[0, W, 0, H],
        cmap="turbo"
    )

    rect = plt.Rectangle(
        (rx, ry),
        rw,
        rh,
        color="black"
    )

    ax.add_patch(rect)

    ax.plot(
        x0,
        y0,
        "wo",
        markersize=8
    )

    ax.set_aspect("equal")

    ax.set_title("Distancia mínima evitando el obstáculo")

    plt.colorbar(
        im,
        ax=ax,
        label="Distancia"
    )

    st.pyplot(fig)