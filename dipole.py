import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from scipy.constants import c, e

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)

# ----------------------------
# 1) Dipol: y-ekseni boyunca salınan +q/-q
# ----------------------------
amplitude = 1e-10        # m
lam = 800e-9             # m  (görsel için dalga boyu seç)
omega = 2.0 * jnp.pi * c / lam
q = e

def r_plus(t):
    return (0.0, +amplitude * jnp.cos(omega * t), 0.0)

def r_minus(t):
    return (0.0, -amplitude * jnp.cos(omega * t), 0.0)

charges = [
    Charge(position_fn=r_plus,  q=+q),
    Charge(position_fn=r_minus, q=-q),
]

print(f"v_max/c = {float(amplitude*omega/c):.3e}")

# ----------------------------
# 2) Grid ve zaman (EM dalgası için birkaç lambda göster)
# ----------------------------
grid_res = 260
grid_extent = 3.0 * float(lam)     # +/- 3 lambda

x = jnp.linspace(-grid_extent, grid_extent, grid_res)
y = jnp.linspace(-grid_extent, grid_extent, grid_res)
z = jnp.array([0.0])

nframes = 80
T0 = 2.0 * jnp.pi / omega
t = jnp.linspace(0.0, T0, nframes)

# Meshgrid: (Nx,Ny,Nz,Nt)
X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# ----------------------------
# 3) Alanları hesapla (tek seferde, sonra animasyonda sadece indexle)
# ----------------------------
quantities_fn = potentials_and_fields(charges)
jit_quantities_fn = jax.jit(quantities_fn)

res = jit_quantities_fn(X, Y, Z, T)

# pycharge field shape'leri sürüme göre bazen (3,...) gelebilir
def to_lastdim3(field):
    if field.shape[-1] == 3:
        return field
    if field.shape[0] == 3:
        return jnp.moveaxis(field, 0, -1)
    raise ValueError(f"Beklenmeyen field shape: {field.shape}")

E = to_lastdim3(res.electric)   # (Nx,Ny,Nz,Nt,3)
B = to_lastdim3(res.magnetic)   # (Nx,Ny,Nz,Nt,3)

# z=0 düzlemini al: (Nx,Ny,Nt,3)
E0 = E[:, :, 0, :, :]
B0 = B[:, :, 0, :, :]

# İstediğimiz bileşenler:
Ey = E0[..., 1]   # (Nx,Ny,Nt)
Bz = B0[..., 2]   # (Nx,Ny,Nt)

# ----------------------------
# 4) SymLog norm (global scale) -> “tek renk” olmaz, hem +/− görünür
# ----------------------------
def sym_lognorm_from(data, linthresh_ratio=1e-5):
    vmax = float(jnp.max(jnp.abs(data)))
    if vmax == 0.0:
        vmax = 1.0
    return colors.SymLogNorm(
        linthresh=vmax * linthresh_ratio,
        linscale=1.0,
        vmin=-vmax,
        vmax=vmax,
        base=10
    )

norm_E = sym_lognorm_from(Ey)
norm_B = sym_lognorm_from(Bz)

extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

# ----------------------------
# 5) Animasyon
# ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

imE = ax1.imshow(
    Ey[:, :, 0].T,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=norm_E,
    aspect="equal",
)
ax1.set_title("E_y (z=0)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
plt.colorbar(imE, ax=ax1, fraction=0.046, pad=0.04, label="E_y")

imB = ax2.imshow(
    Bz[:, :, 0].T,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=norm_B,
    aspect="equal",
)
ax2.set_title("B_z (z=0)")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
plt.colorbar(imB, ax=ax2, fraction=0.046, pad=0.04, label="B_z")

def update(i):
    imE.set_data(Ey[:, :, i].T)
    imB.set_data(Bz[:, :, i].T)
    ax1.set_title(f"E_y (z=0)   t={float(t[i]*1e15):.2f} fs")
    ax2.set_title(f"B_z (z=0)   t={float(t[i]*1e15):.2f} fs")
    return (imE, imB)

ani = FuncAnimation(fig, update, frames=nframes, interval=50, blit=False)
plt.show()

# Kaydetmek istersen:
# ani.save("dipole_EM_wave.gif", writer="pillow", fps=20)