from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
# Reference / validation script for Step11d.
# Used to verify per-slit refinement behavior.
# Not part of pipeline execution.
"""
# ============================================================
# USER INPUT
# ============================================================
slit = "SLIT000"

from path import Path
PATH_REDUCED = Path('/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS')
fn_fits = PATH_REDUCED / "_Run8_Science_2026_01/SAMI/Dolidze25/reduced/11_fluxcal/Extract1D_fluxcal.fits"
fn_phot = PATH_REDUCED / "_Run8_Science_2026_01/SAMI/Dolidze25/reduced/11_fluxcal/slit_trace_radec_skymapper_all.csv"

r_band_file = "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/samos-pipeline/calibration/reference_tables/filters/skymapper_r_nm.txt"
i_band_file = "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/samos-pipeline/calibration/reference_tables/filters/skymapper_i_nm.txt"
z_band_file = "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/samos-pipeline/calibration/reference_tables/filters/skymapper_z_nm.txt"

r_cut_nm = 600.0
z_cut_nm = 930.0

LAM_EFF = {"r": 620.0, "i": 750.0, "z": 870.0}

# ============================================================
# HELPERS
# ============================================================
def trapz(y, x):
    return float(np.trapezoid(y, x))

def load_band(path):
    arr = np.loadtxt(path, comments="#", usecols=(0, 1))
    w = np.asarray(arr[:, 0], float)
    t = np.asarray(arr[:, 1], float)
    o = np.argsort(w)
    return w[o], np.clip(t[o], 0.0, None)

def interp_bandpass(wave_spec_nm, wave_band_nm, tran_band):
    return np.interp(wave_spec_nm, wave_band_nm, tran_band, left=0.0, right=0.0)

def abmag_to_fnu_cgs(mag_ab):
    return 3631.0 * 10**(-0.4 * mag_ab) * 1e-23

def fnu_to_abmag(fnu_cgs):
    return -2.5 * np.log10(fnu_cgs / 3631e-23)

def abmag_to_flam_cgs(mag_ab, lam_nm):
    fnu_cgs = abmag_to_fnu_cgs(mag_ab)
    c_A_s = 2.99792458e18
    lam_A = lam_nm * 10.0
    return float(fnu_cgs * c_A_s / (lam_A**2))

def make_r_short_filter(wr, tr, lam_min=600.0):
    m = wr >= lam_min
    wrs = wr[m].copy()
    trs = tr[m].copy()

    if wrs.size == 0 or wrs[0] > lam_min:
        tmin = np.interp(lam_min, wr, tr, left=0.0, right=0.0)
        wrs = np.insert(wrs, 0, lam_min)
        trs = np.insert(trs, 0, tmin)

    return wrs, trs

def make_z_short_filter(wz, tz, lam_max=930.0):
    m = wz <= lam_max
    wzs = wz[m].copy()
    tzs = tz[m].copy()

    if wzs.size == 0 or wzs[-1] < lam_max:
        tmax = np.interp(lam_max, wz, tz, left=0.0, right=0.0)
        wzs = np.append(wzs, lam_max)
        tzs = np.append(tzs, tmax)

    return wzs, tzs

def effective_lambda_nm(w, t):
    m = np.isfinite(w) & np.isfinite(t) & (t > 0)
    return trapz(w[m] * t[m], w[m]) / trapz(t[m], w[m])

def coverage_fraction(wb, tb, lam_min, lam_max):
    mfull = np.isfinite(wb) & np.isfinite(tb) & (tb > 0)
    mcov = mfull & (wb >= lam_min) & (wb <= lam_max)
    num = trapz(tb[mcov] * wb[mcov], wb[mcov])
    den = trapz(tb[mfull] * wb[mfull], wb[mfull])
    return num / den if den > 0 else np.nan

def predict_fnu_from_photometry(lam_query_nm, lam_pts_nm, mags_ab):
    lam_pts_nm = np.asarray(lam_pts_nm, float)
    mags_ab = np.asarray(mags_ab, float)
    m = np.isfinite(lam_pts_nm) & np.isfinite(mags_ab)
    lam_pts_nm = lam_pts_nm[m]
    mags_ab = mags_ab[m]

    fnu_pts = abmag_to_fnu_cgs(mags_ab)
    logfnu = np.log10(fnu_pts)

    o = np.argsort(lam_pts_nm)
    lam_pts_nm = lam_pts_nm[o]
    logfnu = logfnu[o]

    return 10**np.interp(lam_query_nm, lam_pts_nm, logfnu)

def synth_abmag_from_fnu_model(wband, tband, lam_pts_nm, mags_ab):
    m = np.isfinite(wband) & np.isfinite(tband) & (tband > 0)
    w = np.asarray(wband[m], float)
    t = np.asarray(tband[m], float)

    fnu = predict_fnu_from_photometry(w, lam_pts_nm, mags_ab)

    c_A_s = 2.99792458e18
    w_A = w * 10.0
    flam = fnu * c_A_s / (w_A**2)

    num = trapz(flam * t * w, w)
    den = trapz(t * w, w)
    flam_band = num / den

    lam_eff = effective_lambda_nm(w, t)
    lam_eff_A = lam_eff * 10.0
    fnu_eff = flam_band * lam_eff_A**2 / c_A_s

    return fnu_to_abmag(fnu_eff)

def observed_band_target_ab(mag_ab, wave_nm, tran):
    fnu = abmag_to_fnu_cgs(mag_ab)
    c_ang_s = 2.99792458e18
    wave_ang = wave_nm * 10.0
    flam = fnu * c_ang_s / (wave_ang**2)
    return trapz(flam * tran * wave_nm, wave_nm)

def choose_scaled_coordinate(wave_nm, lambda0_nm=None, scale_nm=None):
    if lambda0_nm is None:
        lambda0_nm = float(0.5 * (np.nanmin(wave_nm) + np.nanmax(wave_nm)))
    if scale_nm is None:
        scale_nm = float(0.5 * (np.nanmax(wave_nm) - np.nanmin(wave_nm)))
    x = (wave_nm - lambda0_nm) / scale_nm
    return x, lambda0_nm, scale_nm

def build_design_row(flux, wave_nm, tran, x):
    w = tran * wave_nm
    return np.array([
        trapz(flux * x**2 * w, wave_nm),
        trapz(flux * x    * w, wave_nm),
        trapz(flux        * w, wave_nm),
    ], dtype=float)

def solve_response(lam, flux, bands, mags):
    x, lambda0_nm, scale_nm = choose_scaled_coordinate(lam)
    rows, rhs, used = [], [], []

    for band in ["r", "i", "z"]:
        mag = mags[band]
        if not np.isfinite(mag):
            continue

        wb, tb = bands[band]
        tran = interp_bandpass(lam, wb, tb)
        if np.nanmax(tran) <= 0:
            continue

        rows.append(build_design_row(flux, lam, tran, x))
        rhs.append(observed_band_target_ab(mag, lam, tran))
        used.append(band)

    M = np.vstack(rows)
    y = np.asarray(rhs, float)

    coeff = np.linalg.solve(M, y)
    a, b, c = coeff
    resp = a*x**2 + b*x + c

    return {
        "coeff": coeff,
        "resp": resp,
        "lambda0_nm": lambda0_nm,
        "scale_nm": scale_nm,
        "used": used,
    }

# ============================================================
# READ STEP11c SPECTRUM
# ============================================================
h = fits.open(fn_fits)
tab = h[slit].data

lam_all = np.asarray(tab["LAMBDA_NM"], float)
flux_all = np.asarray(tab["FLUX_FLAM"], float)

m = np.isfinite(lam_all) & np.isfinite(flux_all)
lam = lam_all[m]
flux = flux_all[m]

o = np.argsort(lam)
lam = lam[o]
flux = flux[o]

print("N finite =", len(lam), " lambda range =", lam.min(), lam.max())

# ============================================================
# READ PHOTOMETRY
# ============================================================
phot = pd.read_csv(fn_phot)
phot["slit"] = phot["slit"].astype(str).str.strip().str.upper()
row = phot.loc[phot["slit"] == slit].iloc[0]

m_r = float(row["r_mag"]) if np.isfinite(row["r_mag"]) else np.nan
m_i = float(row["i_mag"]) if np.isfinite(row["i_mag"]) else np.nan
m_z = float(row["z_mag"]) if np.isfinite(row["z_mag"]) else np.nan

print("Catalog mags:", {"r": m_r, "i": m_i, "z": m_z})

# ============================================================
# LOAD FILTERS
# ============================================================
wr, tr = load_band(r_band_file)
wi, ti = load_band(i_band_file)
wz, tz = load_band(z_band_file)

wrs, trs = make_r_short_filter(wr, tr, lam_min=r_cut_nm)
wzs, tzs = make_z_short_filter(wz, tz, lam_max=z_cut_nm)

print("\nCoverage with observed spectrum:")
print("r  =", coverage_fraction(wr, tr, lam.min(), lam.max()))
print("i  =", coverage_fraction(wi, ti, lam.min(), lam.max()))
print("z  =", coverage_fraction(wz, tz, lam.min(), lam.max()))

print("\nShortened filters:")
print("r_short lambda_eff =", effective_lambda_nm(wrs, trs), " wmin/wmax =", wrs.min(), wrs.max())
print("z_short lambda_eff =", effective_lambda_nm(wzs, tzs), " wmin/wmax =", wzs.min(), wzs.max())

# ============================================================
# DERIVE r_short and z_short MAGNITUDES
# ============================================================
lam_pts = np.array([LAM_EFF["r"], LAM_EFF["i"], LAM_EFF["z"]], float)
mag_pts = np.array([m_r, m_i, m_z], float)

m_rshort = synth_abmag_from_fnu_model(wrs, trs, lam_pts, mag_pts)
m_zshort = synth_abmag_from_fnu_model(wzs, tzs, lam_pts, mag_pts)

print("\nMagnitudes:")
print("r          =", m_r)
print("r_short    =", m_rshort)
print("Δr         =", m_rshort - m_r)
print("z          =", m_z)
print("z_short    =", m_zshort)
print("Δz         =", m_zshort - m_z)

# ============================================================
# SOLVE FULL r/i/z
# ============================================================
bands_full = {
    "r": (wr, tr),
    "i": (wi, ti),
    "z": (wz, tz),
}
mags_full = {
    "r": m_r,
    "i": m_i,
    "z": m_z,
}

sol_full = solve_response(lam, flux, bands_full, mags_full)
resp_full = sol_full["resp"]
flux_full = flux * resp_full

print("\nFULL r/i/z solution")
print("used =", sol_full["used"])
print("coeff =", sol_full["coeff"])
print("median response =", np.nanmedian(resp_full))
print("min/max response =", np.nanmin(resp_full), np.nanmax(resp_full))

# ============================================================
# SOLVE r_short / i / z_short
# ============================================================
bands_short = {
    "r": (wrs, trs),
    "i": (wi, ti),
    "z": (wzs, tzs),
}
mags_short = {
    "r": m_rshort,
    "i": m_i,
    "z": m_zshort,
}

print("\nActual bands passed to SHORT solve:")
for name, (wb, tb) in bands_short.items():
    print(
        name,
        "lambda_eff =", effective_lambda_nm(wb, tb),
        "coverage =", coverage_fraction(wb, tb, lam.min(), lam.max()),
        "wmin/wmax =", wb.min(), wb.max(),
    )

sol_short = solve_response(lam, flux, bands_short, mags_short)
resp_short = sol_short["resp"]
flux_short = flux * resp_short

print("\nR_SHORT / I / Z_SHORT solution")
print("used =", sol_short["used"])
print("coeff =", sol_short["coeff"])
print("median response =", np.nanmedian(resp_short))
print("min/max response =", np.nanmin(resp_short), np.nanmax(resp_short))

# ============================================================
# PLOT 1: r and z filter truncation
# ============================================================
plt.figure(figsize=(8,4))
plt.plot(wr, tr, label="r")
plt.plot(wrs, trs, label=f"r_short (>= {r_cut_nm:.0f} nm)")
plt.plot(wz, tz, label="z")
plt.plot(wzs, tzs, label=f"z_short (<= {z_cut_nm:.0f} nm)")
plt.axvline(r_cut_nm, ls="--", color="k")
plt.axvline(z_cut_nm, ls="--", color="k")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Throughput")
plt.title("Original vs shortened filters")
plt.grid()
plt.legend()
plt.show()

# ============================================================
# PLOT 2: response comparison
# ============================================================
plt.figure(figsize=(7,4))
plt.plot(lam, resp_full, label="full r/i/z", lw=1.5)
plt.plot(lam, resp_short, label="r_short / i / z_short", lw=1.5)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Response")
plt.title(f"{slit}: response comparison")
plt.grid()
plt.legend()
plt.show()

# ============================================================
# PLOT 3: spectra + SkyMapper points
# ============================================================
plt.figure(figsize=(8.5,4.8))
ax = plt.gca()

ax.plot(lam, flux, label="Step11c", lw=0.8)
ax.plot(lam, flux_full, label="full r/i/z refined", lw=1.0)
ax.plot(lam, flux_short, label="r_short/i/z_short refined", lw=1.0)

for band in ["r", "i", "z"]:
    mag = {"r": m_r, "i": m_i, "z": m_z}[band]
    if np.isfinite(mag):
        x0 = LAM_EFF[band]
        y0 = abmag_to_flam_cgs(mag, x0)
        ax.scatter([x0], [y0], s=40, zorder=5)
        ax.text(x0, y0, band, fontsize=9, ha="left", va="bottom")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(r"$f_\lambda$  [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
ax.set_title(slit)
ax.grid()
ax.legend(fontsize=8)
plt.show()