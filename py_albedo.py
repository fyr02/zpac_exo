import fileinput, time
from matplotlib.pyplot import plot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import astropy
from astropy.modeling import models
from astropy import units as u

def bandpass_model(ts, tp, band_wv, band_e):
  b_s = models.BlackBody(temperature=ts*u.K)
  b_p = models.BlackBody(temperature=tp*u.K)
  total_flux_s = 0
  total_flux_b = 0
  for i in range(len(band_wv)):
    w = band_wv[i]
    e = band_e[i]
    total_flux_s = total_flux_s + (b_s(w*u.meter)*e)
    total_flux_b = total_flux_b + (b_p(w*u.meter)*e)
  return (total_flux_b / total_flux_s)

def albedo(d_occ, d_err, rp, rs, ar, ts, btp, band_wv, band_e):
  b_m = bandpass_model(ts, btp, band_wv, band_e)
  a_g = (d_occ*((ar/rp)**2)) - (b_m*(ar**2))
  a_g_e_c = ((d_occ+d_err)*((ar/rp)**2)) - (b_m*(ar**2))
  a_g_err = a_g_e_c - a_g
  return a_g, a_g_err

def albedo_pur(d_occ, d_err, rp, rs, ar, ts, btp, band_wv, band_e):
  a_g = (d_occ*((ar/rp)**2))
  a_g_e_c = ((d_occ+d_err)*((ar/rp)**2))
  a_g_err = a_g_e_c - a_g
  return a_g, a_g_err

def albedo_irr(d_occ, d_err, rp, rs, ar, ts, btp, band_wv, band_e):
  a_g = 0
  f_r = 0.666667
  bxp = ts * ( ar ** -0.5 ) * ( ( f_r * ( 1 - abs( 1.5 * a_g ) ) ) ** 0.25 )
  for i in range(25):
    b_m = bandpass_model(ts, bxp, band_wv, band_e)
    a_g = (d_occ*((ar/rp)**2)) - (b_m*(ar**2))
    bxp = ts * ( ar ** -0.5 ) * ( ( f_r * ( 1 - ( 1.5 * a_g ) ) ) ** -0.25 )
  a_g_e_c = ((d_occ+d_err)*((ar/rp)**2)) - (b_m*(ar**2))
  a_g_err = a_g_e_c - a_g
  return a_g, a_g_err, bxp

def albedo_max(d_occ, d_err, rp, rs, ar, ts, btp, band_wv, band_e):
  f_r = 0.666667
  bxp = ts * ( ar ** -0.5 ) * ( f_r ** 0.25 )
  b_m = bandpass_model(ts, bxp, band_wv, band_e)
  a_g = (d_occ*((ar/rp)**2)) - (b_m*(ar**2))
  a_g_e_c = ((d_occ+d_err)*((ar/rp)**2)) - (b_m*(ar**2))
  a_g_err = a_g_e_c - a_g
  return a_g, a_g_err, bxp

def albedo_hom(d_occ, d_err, rp, rs, ar, ts, btp, band_wv, band_e):
  f_r = 0.25
  bxp = ts * ( ar ** -0.5 ) * ( ( f_r * ( 1 - ( 0.5 ) ) ) ** 0.25 )
  b_m = bandpass_model(ts, bxp, band_wv, band_e)
  a_g = (d_occ*((ar/rp)**2)) - (b_m*(ar**2))
  a_g_e_c = ((d_occ+d_err)*((ar/rp)**2)) - (b_m*(ar**2))
  a_g_err = a_g_e_c - a_g
  return a_g, a_g_err, bxp

def albedo_min(d_occ, d_err, rp, rs, ar, ts, btp, band_wv, band_e):
  a_g = (d_occ*((ar/rp)**2))
  a_g_e_c = ((d_occ+d_err)*((ar/rp)**2))
  a_g_err = a_g_e_c - a_g
  return a_g, a_g_err

filename='database.txt'
filename_f='filter_database.txt'
data = loadtxt(filename, dtype=[('id', 'int'), ('name', 'object'), ('rp', 'float'), ('rs', 'float'), ('a', 'float'), ('ts', 'int'), ('btp', 'int'), ('ref', 'object')])
data_f = loadtxt(filename_f, dtype=[('id', 'int'), ('name', 'object'), ('path', 'object')])
print("input planet ID:")
index=int(input())
ids = data['id']
id_p = np.where(ids==index)
print("input filter ID:")
index=int(input())
ids = data_f['id']
id_f = np.where(ids==index)
name = data[id_p][0][1]
rp = data[id_p][0][2]
rs = data[id_p][0][3]
a = data[id_p][0][4]
ts = data[id_p][0][5]
btp = data[id_p][0][6]
name_f = data_f[id_f][0][1]
filename_bp = data_f[id_f][0][2]
data_b = loadtxt(filename_bp)
b_wv = data_b[:,0]
b_lu = data_b[:,1]


print("input calculation mode - A, B, or C:")
mode=str(input())

#mode = 'A'

print("input eclipse depth (ppm):")
depth=float(input())
print("input error (ppm):")
error=float(input())
d_pm = depth / 1000000
e_pm = error / 1000000


if mode == 'A':
  a_g, a_g_err = albedo(d_pm, e_pm, rp, rs, a, ts, btp, b_wv, b_lu)
  print("Planet: " + name)
  print("Bandpass: " + name_f)
  print("Dayside Temperature: " + str(btp))
  print("Geometric Albedo: " + str(a_g))
  print("Error: " + str(a_g_err))
elif mode == 'B':
  ah_g, ah_g_err, bhp = albedo_hom(d_pm, e_pm, rp, rs, a, ts, btp, b_wv, b_lu)
  am_g, am_g_err, bmp = albedo_max(d_pm, e_pm, rp, rs, a, ts, btp, b_wv, b_lu)
  bnp = 0.5 * (bhp + bmp)
  spread_err = 0.5 * ( am_g - ah_g )
  a_g_err = ( (spread_err ** 2) + (ah_g_err ** 2) ) ** 0.5
  a_g = 0.5 * (ah_g + am_g)
  ffah_g, ffah_g_err, ffbkp = albedo_max(d_pm, e_pm, rp, rs, a, ts, btp, b_wv, b_lu)
  ffam_g, ffam_g_err = albedo_min(d_pm, e_pm, rp, rs, a, ts, btp, b_wv, b_lu)
  ffspread_err = 0.5 * ( ffam_g - ffah_g )
  ffa_g_err = ( (ffspread_err ** 2) + (ffah_g_err ** 2) ) ** 0.5
  ffa_g = 0.5 * (ffah_g + ffam_g)
  print("Planet: " + name)
  print("Bandpass: " + name_f)
  print("Dayside Temperature (mean, optimistic): " + str(bnp))
  print("Geometric Albedo (mean, optimistic  ): " + str(a_g))
  print("Geometric Albedo (mean, conservative): " + str(ffa_g))
  print("Error (mean, optimistic  ): " + str(a_g_err))
  print("Error (mean, conservative): " + str(ffa_g_err))
  print("--------------")
  print("Dayside Temperature (irradiated): " + str(bmp))
  print("Dayside Temperature (homogenous): " + str(bhp))
  print("Dayside Temperature (mean, conservative): " + str(bmp / 2))
  print("Geometric Albedo (no thermal correction): " + str(ffam_g))
  print("Geometric Albedo (homogenous): " + str(ah_g))
  print("Geometric Albedo (max): " + str(ffah_g))
  print("Geometric Albedo - Error (eclipse uncertainty only): " + str(ah_g_err))
else:
  a_g, a_g_err = albedo_pur(d_pm, e_pm, rp, rs, a, ts, btp, b_wv, b_lu)
  print("Planet: " + name)
  print("Bandpass: " + name_f)
  print("Geometric Albedo (no thermal correction): " + str(a_g))
  print("Error: " + str(a_g_err))




