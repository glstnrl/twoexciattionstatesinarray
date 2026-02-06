import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from scipy.constants import c, e, mu_0   # <-- mu_0 eklendi

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)

# ----------------------------
# 1) Parametreler
# ----------------------------
amplitude = 1e-10        # m
lam = 800e-9             # m
omega = 2.0 * jnp.pi * c / lam
q = e

# İki dipolün merkezleri (x yönünde ayrık)
d = lam / 20.0
x0 = -d / 2.0
x1 = +d / 2.0

# Fazlar: in-phase için phi2=0, subradiance eğilimi için phi2=pi
phi1 = 0.0
phi2 = jnp.pi  # <-- 0.0 yaparsan "superradiant/in-phase" gibi

print(f"v_max/c = {float(amplitude*omega/c):.3e}")

# ----------------------------
# 2) İki dipol: toplam 4 charge
# ----------------------------
def r_plus_1(t):
    return (x0, +amplitude * jnp.cos(omega * t + phi1), 0.0)

def r_minus_1(t):
    return (x0, -amplitude * jnp.cos(omega * t + phi1), 0.0)

def r_plus_2(t):
    return (x1, +amplitude * jnp.cos(omega * t + phi2), 0.0)

def r_minus_2(t):
    return (x1, -amplitude * jnp.cos(omega * t + phi2), 0.0)

charges = [
    Charge(position_fn=r_plus_1,  q=+q),
    Charge(position_fn=r_minus_1, q=-q),
    Charge(position_fn=r_plus_2,  q=+q),
    Charge(position_fn=r_minus_2, q=-q),
]

# ----------------------------
# 3) Grid ve zaman (z=0 düzlemi)
# ----------------------------
grid_res = 260
grid_extent = 3.0 * float(lam)

x = jnp.linspace(-grid_extent, grid_extent, grid_res)
y = jnp.linspace(-grid_extent, grid_extent, grid_res)
z = jnp.array([0.0])

nframes = 80
T0 = 2.0 * jnp.pi / omega
t = jnp.linspace(0.0, T0, nframes)

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# ----------------------------
# 4) Alanları hesapla (tek sefer)
# ----------------------------
quantities_fn = potentials_and_fields(charges)
jit_quantities_fn = jax.jit(quantities_fn)
res = jit_quantities_fn(X, Y, Z, T)

def to_lastdim3(field):
    if field.shape[-1] == 3:
        return field
    if field.shape[0] == 3:
        return jnp.moveaxis(field, 0, -1)
    raise ValueError(f"Beklenmeyen field shape: {field.shape}")

E = to_lastdim3(res.electric)
B = to_lastdim3(res.magnetic)

E0 = E[:, :, 0, :, :]   # (Nx,Ny,Nt,3)
B0 = B[:, :, 0, :, :]   # (Nx,Ny,Nt,3)

Ey = E0[..., 1]         # (Nx,Ny,Nt)
Bz = B0[..., 2]         # (Nx,Ny,Nt)

# ----------------------------
# 4.5) POYNTING: S = (E x B)/mu0, |S|
# ----------------------------
S = jnp.cross(E0, B0) / mu_0          # (Nx,Ny,Nt,3)
Smag = jnp.linalg.norm(S, axis=-1)    # (Nx,Ny,Nt)

# ----------------------------
# 5) Renk ölçekleri
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

# |S| pozitif -> LogNorm (pointing_vector örneği gibi)
Smag_max = float(jnp.max(Smag))
Smag_min_pos = float(jnp.min(Smag[Smag > 0])) if jnp.any(Smag > 0) else 1e-30
norm_S = colors.LogNorm(vmin=Smag_min_pos, vmax=Smag_max)

extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

# ----------------------------
# 6) Animasyon (Ey, Bz, |S|)
# ----------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

imE = ax1.imshow(Ey[:, :, 0].T, origin="lower", extent=extent, cmap="RdBu_r", norm=norm_E, aspect="equal")
ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
ax1.scatter([x0, x1], [0.0, 0.0], s=25)
plt.colorbar(imE, ax=ax1, fraction=0.046, pad=0.04, label="E_y")

imB = ax2.imshow(Bz[:, :, 0].T, origin="lower", extent=extent, cmap="RdBu_r", norm=norm_B, aspect="equal")
ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
ax2.scatter([x0, x1], [0.0, 0.0], s=25)
plt.colorbar(imB, ax=ax2, fraction=0.046, pad=0.04, label="B_z")

imS = ax3.imshow(Smag[:, :, 0].T, origin="lower", extent=extent, cmap="inferno", norm=norm_S, aspect="equal")
ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")
ax3.scatter([x0, x1], [0.0, 0.0], s=25)
plt.colorbar(imS, ax=ax3, fraction=0.046, pad=0.04, label=r"$|\mathbf{S}|$ (W/m²)")

def update(i):
    imE.set_data(Ey[:, :, i].T)
    imB.set_data(Bz[:, :, i].T)
    imS.set_data(Smag[:, :, i].T)

    ax1.set_title(f"E_y (z=0)   t={float(t[i]*1e15):.2f} fs   phi2={float(phi2):.2f}")
    ax2.set_title(f"B_z (z=0)   t={float(t[i]*1e15):.2f} fs   phi2={float(phi2):.2f}")
    ax3.set_title(f"|S| (z=0)   t={float(t[i]*1e15):.2f} fs   phi2={float(phi2):.2f}")

    return (imE, imB, imS)

ani = FuncAnimation(fig, update, frames=nframes, interval=50, blit=False)
plt.show()

# Kaydetmek istersen:
# ani.save("two_dipole_EB_S.gif", writer="pillow", fps=20)