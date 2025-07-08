###############################################################################
#                           USER INPUT — edit these                           #
###############################################################################
# ── Geometry (mm) ────────────────────────────────────────────────────────────
B_mm        = 210.0   # outer dimension
D_outer_mm  = 140.0   # steel tube outer diameter
t_mm        =   4.8   # tube thickness  (set 0 for solid tube)

# Rebars (assume 4 bars at corners)
cover_mm      = 40.0   # concrete cover to bar centre
rebar_diam_mm = 10.0   # bar Ø  (all identical)

# ── Material moduli (GPa) ────────────────────────────────────────────────────
E_c_out_GPa = 32.5   # OUTER concrete encasement
E_c_in_GPa  = 39.5   # INNER concrete core (inside tube) – could differ
E_s_GPa     = 200.0  # steel tube **and** rebars

# ── Column & analysis parameters ────────────────────────────────────────────
M_u_kNm   = 59.0   # ultimate moment (kN·m) calculation refer to Ma and Tan (2022)
e_mm      =  51.0   # load eccentricity (mm)
L_e_mm    = 2150.0   # effective length (mm)
dN_kN     =   1.0   # load increment (kN)
###############################################################################
#                   No need to edit anything below this line
###############################################################################

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 2. Compute elastic EI_el  (outer conc. + tube steel + core conc. + rebars)
# ---------------------------------------------------------------------------
def elastic_EI(B, Do, t, Ec_out, Ec_in, Es, cov, phi):
    """
    Composite EI_el [kN·m²] for a square-encased CFST with 4 corner rebars.
    All lengths in metres; moduli in kN/m².
    """
    Di = Do - 2*t
    if Di < 0:
        raise ValueError("Tube thickness too large (inner Ø < 0).")

    # ----- geometric inertias (m⁴) -----
    I_square = B**4 / 12
    I_outer  = np.pi * Do**4 / 64
    I_inner  = np.pi * Di**4 / 64

    # Rebars:
    A_bar   = np.pi * phi**2 / 4
    I_bar_c = np.pi * phi**4 / 64          # intrinsic about bar centre
    x = y = B/2 - cov - phi/2
    r_sq = x**2 + y**2
    I_bars_total = 4 * (I_bar_c + A_bar * r_sq)

    # Concrete parts (remove bar cavities from shell)
    I_conc_out = I_square - I_outer - I_bars_total   # outer concrete
    I_conc_in  = I_inner                             # inner concrete
    I_tube     = I_outer - I_inner                   # steel tube

    # Composite EI
    EI = (Ec_out * I_conc_out +
          Ec_in  * I_conc_in  +
          Es     * (I_tube + I_bars_total))

    return EI  # kN·m²

# Convert to metres & kN/m²
B, Do, t = [x/1_000 for x in (B_mm, D_outer_mm, t_mm)]
cov, phi = [x/1_000 for x in (cover_mm, rebar_diam_mm)]
Ec_out = E_c_out_GPa * 1_000_000
Ec_in  = E_c_in_GPa  * 1_000_000
Es     = E_s_GPa     * 1_000_000

EI_el = elastic_EI(B, Do, t, Ec_out, Ec_in, Es, cov, phi)
print(f"Computed EI_el = {EI_el:,.2f} kN·m²")

# ---------------------------------------------------------------------------
# 3. Analytical model (core algorithm)
# ---------------------------------------------------------------------------
class CECFSTColumn:
    def __init__(self, EI_el, M_u, e, L_e, *, dN=1.0):
        self.EI_el = EI_el
        self.M_u   = M_u
        self.e     = e                 # m
        self.L_e   = L_e               # m
        self.dN    = dN
        self.omega   = L_e / 1_000
        self.kappa_u = 2*M_u / EI_el

    def _EI(self, k): return self.EI_el * (1 - k/self.kappa_u)
    def _step_asc(self, N, M, k, d):
        Nn = N + self.dN
        Mn = Nn * (self.e + self.omega + d/1_000)
        kn = k + (Mn - M) / self._EI(k)
        dn = self.L_e**2 * kn / 8 * 1_000
        return Nn, Mn, kn, dn
    def _step_desc(self, N, d_prev):
        Nn = N - self.dN
        if Nn <= 0: return None, None
        C = self.e + self.omega
        dn = (N/Nn)*(C + d_prev/1_000) - C
        return Nn, dn*1_000

    def run(self, *, max_steps=10000):
        stop_ratio = 0.70   # fixed 70 % of peak load
        N = M = k = d = 0.0
        Ns, ds = [], []
        for _ in range(max_steps):           # ascending
            N, M, k, d = self._step_asc(N, M, k, d)
            Ns.append(N); ds.append(d)
            if k >= self.kappa_u:
                N_peak, d_peak = N, d
                break
        cut = stop_ratio*N_peak
        for _ in range(max_steps):           # descending
            N, d = self._step_desc(N, d)
            if N is None: break
            Ns.append(N); ds.append(d)
            if N <= cut: break
        return np.array(Ns), np.array(ds), N_peak, d_peak

# ---------------------------------------------------------------------------
# 4. Run analysis
# ---------------------------------------------------------------------------
e_m, L_e_m = e_mm/1_000, L_e_mm/1_000
col = CECFSTColumn(EI_el, M_u_kNm, e_m, L_e_m, dN=dN_kN)
loads, defls, N_peak, d_peak = col.run()

print(f"\nPeak load  N_peak  = {N_peak:.2f} kN")
print(f"Deflection at peak = {d_peak:.3f} mm")
print(f"Stored points      = {len(loads)}")

# ---------------------------------------------------------------------------
# 5. Plot N–δ curve
# ---------------------------------------------------------------------------
plt.figure(figsize=(5,4))
plt.plot(defls, loads, label="N–δ curve")
plt.plot(d_peak, N_peak, 'ro', label="Peak")
plt.xlabel("Deflection δ (mm)")
plt.ylabel("Axial load N (kN)")
plt.title("CECFST column response"); plt.grid(alpha=.3); plt.legend()
plt.tight_layout(); plt.show()
