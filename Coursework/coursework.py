#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified OFDM code with SER and BER calculation over varying SNR (optimized plotting).
"""
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

# Параметры OFDM
T = 1e-4  # Длительность символа
Nc = 100000   # Количество поднесущих
ts = T / Nc  # Интервал дискретизации
t = ts * np.arange(0, Nc)  # Массив времени для поднесущих

def qam_demodulate(symbols):
    bits_real = (np.real(symbols) > 0).astype(int)
    bits_imag = (np.imag(symbols) > 0).astype(int)
    return np.column_stack((bits_real, bits_imag)).flatten()

def generate_ofdm_data():
    sd_start = np.sign(np.random.rand(1, Nc) - 0.5) + 1j * np.sign(np.random.rand(1, Nc) - 0.5)
    pilot_index = np.arange(0, Nc, 8)
    pilot_value = (1 + 1j) * np.ones(len(pilot_index))
    sd = np.zeros(Nc, dtype=complex)
    data_index = np.setdiff1d(np.arange(Nc), pilot_index)
    sd[pilot_index] = pilot_value
    sd[data_index] = sd_start.flatten()[:len(data_index)]
    return sd, sd_start, pilot_index, pilot_value, data_index

def add_awgn(signal, snr_db):
    snr_linear = 10**(snr_db / 10)
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

snr_values_db = np.arange(0, 21, 2)
ber_values = []

tx_signals = []
rx_signals = []
corrected_symbols = []
channel_estimates = []

for snr_db in snr_values_db:
    # Генерация данных OFDM
    sd, sd_start, pilot_index, pilot_value, data_index = generate_ofdm_data()
    nTap = 3
    ht = np.array([0.7, 0.2, 0.1])
    tx_signal = np.fft.ifft(sd, len(sd))
    tx_signal = np.concatenate((tx_signal[-16:], tx_signal))
    # Сохраняем передаточный сигнал
    tx_signals.append(tx_signal)
    tx_signal = np.convolve(ht, tx_signal)
    # Передача через канал с шумом
    rx_signal = add_awgn(tx_signal, snr_db)
    rx_signal_no_cp = rx_signal[16:Nc+16]

    received_data = np.fft.fft(rx_signal_no_cp, Nc)
    rx_signals.append(received_data)
    
    # Коррекция канала
    HLS = received_data[pilot_index] / pilot_value
    Hc = np.interp(np.arange(Nc), pilot_index, HLS)
    # channel_estimates.append(Hc)
        
    corrected_data = received_data / Hc
    corrected_symbols.append(corrected_data[data_index])

    # BER
    original_symbols = sd_start.flatten()[:len(data_index)]
    received_symbols = corrected_data[data_index]
    transmitted_bits = qam_demodulate(original_symbols)
    received_bits = qam_demodulate(received_symbols)
    bit_errors = np.sum(transmitted_bits != received_bits)
    ber = bit_errors / len(transmitted_bits)
    ber_values.append(ber)

# Визуализация итоговых графиков
plt.figure(figsize=(10, 6))
plt.scatter(rx_signals[-1].real, rx_signals[-1].imag)
plt.title("Сравнение QAM символов до коррекции")
plt.xlabel("Re")
plt.ylabel("Im")
plt.legend()
plt.grid()

plt.figure(figsize=(10, 6))
plt.scatter(corrected_symbols[-1].real, corrected_symbols[-1].imag, label=f"Corrected, SNR={snr_values_db[-1]} dB")
plt.title("Сравнение QAM символов после коррекции")
plt.xlabel("Re")
plt.ylabel("Im")
plt.legend()
plt.grid()

plt.figure(figsize=(10, 6))
plt.semilogy(snr_values_db, ber_values, 's-', label="BER (битовые ошибки)", color='blue')
plt.title("Зависимость BER от SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("BER (логарифмическая шкала)")
plt.legend()
plt.grid()

plt.show()
