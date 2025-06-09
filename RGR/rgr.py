#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:41:07 2024

@author: mikhailkatsuro
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
import numpy as np
from numpy.linalg import inv, svd
from numpy import linalg as la


def H_LS(yn, Xp, pilot_loc, Nfft, Nps):
    Np = Nfft/Nps
    LS = np.zeros(48, dtype=complex)
    for k in range(0, 24):
        LS[(pilot_loc[k]).astype(int)] = yn[(pilot_loc[k]).astype(int)]/srs[k]
    return LS

Hmatr = sp.loadmat("/Users/mikhailkatsuro/Downloads/4 курс/kalachikov/H16_4_35.mat") #16 антенн на БС, 400 поднесущих в полосе частот 20 МГц

hl = list(Hmatr.values())
h = hl[3]

srs24 = sp.loadmat("/Users/mikhailkatsuro/Downloads/4 курс/kalachikov/srs24.mat") #загружаем SRS последовательность
srs1 = list(srs24.values())
srs = srs1[3]

h1 = h[:,0,0] #вектор канала пользователя 1 на поднесущей 1
h2 = h[:,0,1] #вектор канала пользователя 2 на поднесущей 1
h3 = h[:,0,2] #вектор канала пользователя 3 на поднесущей 1
h4 = h[:,0,3] #вектор канала пользователя 4 на поднесущей 1


h1 = h1/la.norm(h1)
h2 = h2/la.norm(h2)
h3 = h3/la.norm(h3)
h4 = h4/la.norm(h4)

H = np.zeros([16,4],dtype = 'complex_') #матрица канала пользователей
W = np.zeros([16,4],dtype = 'complex_') #матрица векторов прекодеров
W0 = np.zeros([16,4],dtype = 'complex_')# нормированная матрица векторов

H[:,0] = h1
H[:,1] = h2
H[:,2] = h3
H[:,3] = h4

H = H.T
H1 = H.conj().T
H2 = H@H1
W = H1@inv(H2) #вычисление матрицы прекодирования

U, S, Vh = svd(H)
rank = np.sum(S > 1e-10)
print(f"Ранг матрицы Н = {rank}")

opimate = np.max(S)
print(f"Mаксимальное сингулярное число = {opimate}")
#w1=w1/la.norm(w1)
#w2=w2/la.norm(w2)
#w3=w3/la.norm(w3)
#w4=w4/la.norm(w4)

w1 = W[:,0]
w2 = W[:,1]
w3 = W[:,2]
w4 = W[:,3]

r = h2.T@w1 #проверка подавления интерференции для канала 2 вектором w1
print(r)

W0[:,0] = w1;
W0[:,1] = w2;
W0[:,2] = w3;
W0[:,3] = w4;

x1 = 1 + 3j
x2 = -3 + 5j
x3 = 5 + 2j
x4 = 9 - 3j

x_t = x1 * w1 + x2 * w2
x_tt = x3 * w3 + x4 * w4

noice = np.random.normal(0, 0.5, len(x_t)) + 1j * np.random.normal(0, 0.5, len(x_t))
noice2 = np.random.normal(0, 0.5, len(x_tt)) + 1j * np.random.normal(0, 0.5, len(x_tt))

x_t1 = x_t + noice
x_t2 = x_tt + noice2

y1 = h1 @ x_t1
print(y1)
y2 = h2 @ x_t1
print(y2)
y3 = h3 @ x_t2
print(y3)
y4 = h4 @ x_t2
print(y4)

Ik=H@W0 #произведение матрицы канала пользователей на матрицу прекодирования
print(Ik)

Nfft = 48
Nps = 2
ip = 0
pilot_loc = np.zeros(24)
td = np.ones(48, dtype = complex)
y = np.zeros(48, dtype = complex)

for k in range (0, Nfft):
    if k%Nps == 1:
        td[k] = srs[ip]
pilot_loc[ip] = k
ip = ip+1

h1 = h[0,300:348,0]
y = td*h1


y2 = y**2
y_db = 10*np.log10(y2)
snr_db = 10
sig_awg_watts = np.mean(y2)
sig_awg_db = 10*np.log10(sig_awg_watts)
noise_awg_db = sig_awg_db - snr_db
noise_awg_watts = 10**(noise_awg_db /10)
mean_noise = 0
noise = np.random.normal(mean_noise, np.sqrt(noise_awg_watts), len(y2))
yn = y+noise

# Вычисляем оценку канала LS
hLS = H_LS(yn, srs, pilot_loc, Nfft, Nps)

gain_matrix_zf = np.abs(H@W0)**2
temp_zf = 1 + np.sum(gain_matrix_zf, 1)
temp_zf = np.diag(temp_zf) * np.ones(4)
int_power_zf = temp_zf - gain_matrix_zf 
SINR_matrix_zf = gain_matrix_zf / int_power_zf
rate_zf = np.sum(np.log2(1+snr_db*np.diag(SINR_matrix_zf)))

print(f"Спектральная эффективность ZF: {rate_zf: f} бит/с")

# Константы
c = 3e8  # скорость света
f = 2.5e9  # частота
lam = c / f  # длина волны
d_x = d_y = lam / 2  # расстояние между элементами
N_x = 4  # количество элементов в направлении X
N_y = 4  # количество элементов в направлении Y
k =  2*np.pi / lam  # волновое число

# Углы наблюдения
theta = np.linspace(-np.pi, np.pi, 360)  # диапазон углов θ
phi = np.linspace(-np.pi, np.pi, 360)  # диапазон углов φ
theta_grid, phi_grid = np.meshgrid(theta, phi)

# Весовой вектор
weights = w1.reshape(N_x, N_y)

# Расчёт AF (диаграммы направленности)
AF = np.zeros_like(theta_grid, dtype=complex)

for m in range(N_x):
    for n in range(N_y):
        phase_shift = (
            (m - 1) * k * d_x * np.sin(theta_grid) * np.cos(phi_grid) +
            (n - 1) * k * d_y * np.sin(theta_grid) * np.sin(phi_grid)
        )
        AF += weights[m, n] * np.exp(1j * phase_shift)

# Усреднение и нормализация
AF_amplitude = np.abs(AF)**2
# AF_amplitude /= np.max(AF_amplitude)

# Построение диаграммы направленности
plt.figure(figsize=(10, 8))
plt.polar(phi, AF_amplitude[int(theta_grid.shape[0] / 2) - 1], label=f"θ = {np.rad2deg(theta[int(theta_grid.shape[0] / 2) - 1])}°")
plt.title("Диаграмма направленности планарной антенной решетки", va="bottom")
plt.legend()
plt.show()

# 3D-график диаграммы направленности
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
x = AF_amplitude * np.sin(theta_grid) * np.cos(phi_grid)
z = AF_amplitude * np.sin(theta_grid) * np.sin(phi_grid)
y = AF_amplitude * np.cos(theta_grid)

ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.set_xlim([-300, 300])
ax.set_ylim([-300, 300])
ax.set_zlim([-300, 300])
ax.set_title("3D-диаграмма направленности планарной антенной решетки")
plt.show()

