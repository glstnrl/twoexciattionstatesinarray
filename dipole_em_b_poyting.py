import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import c, e, mu_0

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)

# ----------------------------
# 1) Kaynaktaki örnekle aynı parametre seçimi
# ----------------------------
amplitude = 5e-11                 # 50 pm  (örnekte böyle)   [oai_citation:2‡pycharge.readthedocs.io](https://pycharge.readthedocs.io/en/stable/examples/pointing_vector.html)
omega = (0.8 * c) / amplitude     # vmax ~ 0.8c             [oai_citation:3‡pycharge.readthedocs.io](https://pycharge.readthedocs.io/en/stable/examples/pointing_vector.html)
charge_magnitude = e

# İki dipol arası mesafe (subradiance görmek için genelde λ'dan küçük seçilir)
# Burada sadece "örnek iskeleti" tuttuğumuz için nm ölçeğinde veriyoruz.
d = 2.0e-10  # 0.2 nm

# Fazlar
phi1 = 0.0
phi2 = jnp.pi   # <-- 0.0 yaparsan in-phase

# ----------------------------
# 2) İki dipol: toplam 4 charge
# Dipoller x ekseninde; her dipol kendi ekseni boyunca (x) salınıyor (örnekle aynı)
# ----------------------------
def pos_plus_1(t):
    return (-d/2 + amplitude * jnp.sin(omega * t + phi1), 0.0, 0.0)

def pos_minus_1(t):
    return (-d/2 - amplitude * jnp.sin(omega * t + phi1), 0.0, 0.0)

def pos_plus_2(t):
    return (+d/2 + amplitude * jnp.sin(omega * t + phi2), 0.0, 0.0)

def pos_minus_2(t):
    return (+d/2 - amplitude * jnp.sin(omega * t + phi2), 0.0, 0.0)

charges = [
    Charge(position_fn=pos_plus_1,  q=+charge_magnitude),
    Charge(position_fn=pos_minus_1, q=-charge_magnitude),
    Charge(position_fn=pos_plus_2,  q=+charge_magnitude),
    Charge(position_fn=pos_minus_2, q=-charge_magnitude),
]

print(f"v_max/c = {float(amplitude * omega / c):.3f}")

# ----------------------------
# 3) Grid (örnekle aynı mantık)
# ----------------------------
grid_res = 800
grid_extent = 1e-9  # ±1 nm   [oai_citation:4‡pycharge.readthedocs.io](https://pycharge.readthedocs.io/en/stable/examples/pointing_vector.html)

x = jnp.linspace(-grid_extent, grid_extent, grid_res)
y = jnp.linspace(-grid_extent, grid_extent, grid_res)
z = jnp.array([0.0])

# t=0 snapshot (örnekte böyle)  [oai_citation:5‡pycharge.readthedocs.io](https://pycharge.readthedocs.io/en/stable/examples/pointing_vector.html)
# B zayıf görünürse bunu aç:
# t = jnp.array([jnp.pi / (2.0 * omega)])
t = jnp.array([0.0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# ----------------------------
# 4) Alanlar + Poynting (örnektekiyle aynı)
# ----------------------------
calculate_fields = jax.jit(potentials_and_fields(charges))   #  [oai_citation:6‡pycharge.readthedocs.io](https://pycharge.readthedocs.io/en/stable/examples/pointing_vector.html)
result = calculate_fields(X, Y, Z, T)

electric_field = result.electric.squeeze()   #  [oai_citation:7‡pycharge.readthedocs.io](https://pycharge.readthedocs.io/en/stable/examples/pointing_vector.html)
magnetic_field = result.magnetic.squeeze()   #  [oai_citation:8‡pycharge.readthedocs.io](https://pycharge.readthedocs.io/en/stable/examples/pointing_vector.html)

extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

poynting_vector = jnp.cross(electric_field, magnetic_field, axis=-1) / mu_0  #  [oai_citation:9‡pycharge.readthedocs.io](https://pycharge.readthedocs.io/en/stable/examples/pointing_vector.html)
poynting_magnitude = jnp.linalg.norm(poynting_vector, axis=-1)               #  [oai_citation:10‡pycharge.readthedocs.io](https://pycharge.readthedocs.io/en/stable/examples/pointing_vector.html)

# ----------------------------
# 5) Plot: |S|
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    poynting_magnitude.T,
    origin="lower",
    cmap="inferno",
    extent=extent,
    # örnekte sabit vmax veriyor; iki dipolde değer değişebilir:
    vmax=float(jnp.quantile(poynting_magnitude, 0.999)),
)
fig.colorbar(im, ax=ax, label=r"$|\mathbf{S}|$ (W/m²)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"Two Dipoles: Poynting Magnitude  (phi2={float(phi2):.2f} rad)")
fig.tight_layout()
plt.show()