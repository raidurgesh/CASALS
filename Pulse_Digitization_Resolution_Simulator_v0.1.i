#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:14:00 2025

@author: drai

Pulse digitization & resolution simulator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import os
import datetime

# Configuration

# Pulse parameters
PULSE_FWHM = 2e-9          # 2 ns FWHM
PULSE_AMP = 1.0            # amplitude of each pulse

# Timing
BASE_T0 = 50e-9            # first pulse center time (s)
GAPS_TO_TEST = np.array([0.5e-9, 1e-9, 2e-9, 5e-9, 10e-9, 20e-9])  # gaps (s)

# Simulation time and high-resolution sampling
SIM_TIME = 200e-9          # total sim window (s)
FS_HIGH = 50e9             # internal high-res sampling (Hz)
T = np.arange(0, SIM_TIME, 1.0/FS_HIGH)

# Noise + jitter
NOISE_RMS = 0.02          # additive white noise RMS (relative to pulse amplitude)
JITTER_STD = 30e-12       # RMS timing jitter per pulse (s)

# Monte-Carlo
MONTE_CARLO_RUNS = 200    # trials per config (increase for higher statistical confidence)

# Rise times to test (s) -> gives different front-end bandwidths
RISE_TIMES = np.array([0.2e-9, 0.5e-9, 1e-9, 2e-9])

# Extra LPF (None to skip)
EXTRA_LPF_FC = None  # e.g., 500e6

# ADC configs: list of tuples (sample_rate_Hz, bits)
ADC_CONFIGS = [
    (250e6, 12),
    (1e9, 12),
    (2e9, 12),
    (5e9, 10),
]

# FFT settings for spectral plots
FFT_LEN = 2**14

# Output paths
OUT_DIR = "."
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUT = os.path.join(OUT_DIR, f"digitization_resolution_summary_{TIMESTAMP}.csv")

# Reproducible randomness seed
SEED = 12345
rng = np.random.default_rng(SEED)

# Functions

def gaussian_pulse(t, t0, fwhm, amplitude=1.0):
    """Gaussian pulse with specified FWHM (seconds)."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amplitude * np.exp(-0.5 * ((t - t0)/sigma)**2)

def rc_impulse_response(t_vec, fc):
    """Single-pole RC impulse response (h(t) = 1/RC * exp(-t/RC) for t>=0)."""
    if fc is None or fc <= 0:
        return np.zeros_like(t_vec)
    RC = 1.0 / (2.0 * np.pi * fc)
    h = np.zeros_like(t_vec)
    mask = (t_vec >= 0)
    h[mask] = (1.0/RC) * np.exp(-t_vec[mask]/RC)
    return h

def fft_convolve(x, h):
    """FFT-based convolution returning same length as x (assumes x length >= h length)."""
    n = len(x) + len(h) - 1
    X = np.fft.rfft(x, n=n)
    H = np.fft.rfft(h, n=n)
    y = np.fft.irfft(X * H, n=n)
    return y[:len(x)]

def measure_fwhm_highres(tvec, yvec):
    """Measure FWHM from high-res vectors (linear interp at half max)."""
    peak = np.max(yvec)
    if peak <= 0:
        return np.nan
    half = peak / 2.0
    above = yvec >= half
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]
    left_idx, right_idx = idx[0], idx[-1]
    # interp left
    if left_idx == 0:
        t_left = tvec[0]
    else:
        y0, y1 = yvec[left_idx-1], yvec[left_idx]
        t_left = tvec[left_idx-1] + (half - y0) * (tvec[left_idx] - tvec[left_idx-1]) / (y1 - y0)
    # interp right
    if right_idx == len(yvec)-1:
        t_right = tvec[-1]
    else:
        y0, y1 = yvec[right_idx], yvec[right_idx+1]
        t_right = tvec[right_idx] + (half - y0) * (tvec[right_idx+1] - tvec[right_idx]) / (y1 - y0)
    return t_right - t_left

def measure_rise_time_10_90(tvec, yvec):
    """10%-90% rise time on the first rising edge found in yvec."""
    ymin, ymax = np.min(yvec), np.max(yvec)
    thr10 = ymin + 0.1 * (ymax - ymin)
    thr90 = ymin + 0.9 * (ymax - ymin)
    # find indices where signal crosses 10% and 90% in rising portion
    above10 = np.where(yvec >= thr10)[0]
    above90 = np.where(yvec >= thr90)[0]
    if len(above10) == 0 or len(above90) == 0:
        return np.nan
    i10, i90 = above10[0], above90[0]
    # linear interpolation helpers
    def interp(i_prev, i_next, target):
        y0, y1 = yvec[i_prev], yvec[i_next]
        if y1 == y0: return tvec[i_prev]
        return tvec[i_prev] + (target - y0) * (tvec[i_next] - tvec[i_prev]) / (y1 - y0)
    t10 = interp(i10-1, i10, thr10) if i10 > 0 else tvec[0]
    t90 = interp(i90-1, i90, thr90) if i90 > 0 else tvec[0]
    return t90 - t10

def adc_sample_and_quantize(t_high, y_high, adc_rate, bits, rng_phase=True, rng=None):
    """Sample the high-res signal at adc_rate and quantize to 'bits'. Returns (ts, ys, ys_q)."""
    dt = 1.0 / adc_rate
    if rng_phase:
        phase = rng.uniform(0, dt) if rng is not None else np.random.uniform(0, dt)
    else:
        phase = 0.0
    ts = np.arange(0, t_high[-1], dt) + phase
    ts = ts[ts < t_high[-1]]
    if ts.size == 0:
        return ts, np.array([]), np.array([])
    ys = np.interp(ts, t_high, y_high)
    # Choose full-scale slightly above peak to avoid clipping
    fs = np.max(np.abs(y_high)) * 1.1 + 1e-12
    q_levels = 2**bits
    # map to signed integer range [-2^(bits-1), 2^(bits-1)-1]
    ints = np.round((ys / fs) * (q_levels/2 - 1))
    ints = np.clip(ints, -q_levels/2, q_levels/2 - 1)
    ys_q = ints * (fs / (q_levels/2 - 1))
    return ts, ys, ys_q

# Sweep

results = []

# Precompute FFT freq axis
freqs = np.fft.rfftfreq(FFT_LEN, d=1.0/FS_HIGH)

for rise_time in RISE_TIMES:
    # approximate front-end BW from rise time: BW ≈ 0.35 / trise
    frontend_bw = 0.35 / rise_time
    # build impulse responses anchored at t=0
    h_front = rc_impulse_response(T - T[0], frontend_bw)
    h_extra = rc_impulse_response(T - T[0], EXTRA_LPF_FC) if EXTRA_LPF_FC else None

    for gap in GAPS_TO_TEST:
        # reset counters/accumulators for Monte Carlo statistics
        resolved_count = {cfg: 0 for cfg in ADC_CONFIGS}
        timing_errs = {cfg: [] for cfg in ADC_CONFIGS}
        amp_errs = {cfg: [] for cfg in ADC_CONFIGS}
        fwhm_values = []
        rise10_90_values = []

        # Monte Carlo runs
        for run in range(MONTE_CARLO_RUNS):
            # jitter each pulse center
            t0_1_j = BASE_T0 + rng.normal(0.0, JITTER_STD)
            t0_2_j = BASE_T0 + gap + rng.normal(0.0, JITTER_STD)
            # create high-res pulse train
            sig = (gaussian_pulse(T, t0_1_j, PULSE_FWHM, PULSE_AMP)
                   + gaussian_pulse(T, t0_2_j, PULSE_FWHM, PULSE_AMP))
            # apply frontend
            sig_front = fft_convolve(sig, h_front)
            if h_extra is not None:
                sig_front = fft_convolve(sig_front, h_extra)
            # add noise
            sig_noisy = sig_front + rng.normal(0.0, NOISE_RMS, size=sig_front.shape)

            # measure high-res metrics (deterministic for given jittered run)
            # restrict to a window around pulses for metrics
            mask_win = (T > BASE_T0 - 10e-9) & (T < BASE_T0 + gap + 10e-9)
            if np.any(mask_win):
                fwhm_val = measure_fwhm_highres(T[mask_win], sig_front[mask_win])
                r10_90 = measure_rise_time_10_90(T[mask_win], sig_front[mask_win])
            else:
                fwhm_val = np.nan
                r10_90 = np.nan
            fwhm_values.append(fwhm_val)
            rise10_90_values.append(r10_90)

            # get true high-res peak for pulse1 (for amplitude comparisons)
            hr_mask_p1 = (T > t0_1_j - 5e-9) & (T < t0_1_j + 5e-9)
            true_peak_p1 = np.max(sig_front[hr_mask_p1]) if np.any(hr_mask_p1) else np.nan

            # ADC sampling & analysis per config
            for cfg in ADC_CONFIGS:
                adc_rate, bits = cfg
                ts_adc, ys_adc, ys_q = adc_sample_and_quantize(T, sig_noisy, adc_rate, bits, rng_phase=True, rng=rng)
                if ys_adc.size == 0:
                    continue
                # search window around pulses
                mask_search = (ts_adc > BASE_T0 - 10e-9) & (ts_adc < BASE_T0 + gap + 10e-9)
                if not np.any(mask_search):
                    # no samples in window (very low ADC rate or unlucky phase)
                    continue
                ts_win = ts_adc[mask_search]
                ys_win = ys_adc[mask_search]
                # find peaks in the sampled window
                peaks, props = signal.find_peaks(ys_win, height=0.1*PULSE_AMP)
                # If less than 2 peaks, not resolved; if >=2, select two highest
                if peaks.size < 2:
                    resolved = False
                else:
                    # pick two highest peaks by height
                    peak_heights = ys_win[peaks]
                    top_idx = np.argsort(peak_heights)[-2:]
                    p1_idx, p2_idx = peaks[top_idx]
                    t_peaks = np.sort(np.array([ts_win[p1_idx], ts_win[p2_idx]]))
                    measured_sep = t_peaks[1] - t_peaks[0]
                    # use high-res FWHM (median of run's fwhm_val approximated by fwhm_val)
                    # resolution criterion: separation > 0.5 * (FWHM1 + FWHM2) ≈ FWHM_val (symmetric assumption)
                    resolved = False
                    if not np.isnan(fwhm_val):
                        resolved = (measured_sep > 0.5 * (fwhm_val + fwhm_val))
                    # timing error relative to true t0_1_j
                    timing_errs[cfg].append(t_peaks[0] - t0_1_j)
                    # amplitude error relative to true high-res peak
                    samp_peak = ys_win[peaks[top_idx[-1]]] if peaks.size>0 else np.nan
                    if not np.isnan(true_peak_p1) and true_peak_p1 != 0:
                        amp_errs[cfg].append((samp_peak - true_peak_p1) / true_peak_p1)
                    else:
                        amp_errs[cfg].append(np.nan)
                if resolved:
                    resolved_count[cfg] += 1

        # aggregate results for this configuration (rise_time, gap)
        for cfg in ADC_CONFIGS:
            adc_rate, bits = cfg
            frac_resolved = resolved_count[cfg] / MONTE_CARLO_RUNS
            median_fwhm = np.nanmedian(fwhm_values)
            median_rise10_90 = np.nanmedian(rise10_90_values)
            timing_errs_arr = np.array(timing_errs[cfg]) if len(timing_errs[cfg])>0 else np.array([])
            amp_errs_arr = np.array(amp_errs[cfg]) if len(amp_errs[cfg])>0 else np.array([])

            results.append({
                'timestamp': TIMESTAMP,
                'rise_time_ns': rise_time * 1e9,
                'frontend_BW_GHz': (0.35 / rise_time) / 1e9,
                'gap_ns': gap * 1e9,
                'adc_rate_GSPS': adc_rate / 1e9,
                'adc_bits': bits,
                'resolved_fraction': frac_resolved,
                'median_fwhm_ns': median_fwhm * 1e9 if not np.isnan(median_fwhm) else np.nan,
                'median_rise10_90_ns': median_rise10_90 * 1e9 if not np.isnan(median_rise10_90) else np.nan,
                'timing_error_rms_ps': np.nanstd(timing_errs_arr) * 1e12 if timing_errs_arr.size>0 else np.nan,
                'amplitude_error_rms_pct': np.nanstd(amp_errs_arr) * 100.0 if amp_errs_arr.size>0 else np.nan
            })

# Results

df = pd.DataFrame(results)
df = df.sort_values(['rise_time_ns','gap_ns','adc_rate_GSPS'])


# Plots

# 1) Resolved fraction vs gap for each frontend BW and ADC config
plt.figure(figsize=(10,6))
for (bw, grp_bw) in df.groupby('frontend_BW_GHz'):
    for adc, grp in grp_bw.groupby('adc_rate_GSPS'):
        label = f"{bw:.2f} GHz BW / {adc:.1f} GS/s"
        plt.plot(grp['gap_ns'], grp['resolved_fraction'], marker='o', label=label)
plt.xlabel("Pulse gap (ns)")
plt.ylabel("Resolved fraction (Monte Carlo)")
plt.title("Probability of resolving two 2 ns pulses vs gap")
plt.grid(True)
plt.legend(fontsize='small', ncol=2)
plt.tight_layout()

# 2) FWHM vs frontend bandwidth for a representative gap (choose 5 ns if available)
rep_gap_ns = 5.0
plt.figure(figsize=(8,5))
sel = df[np.isclose(df['gap_ns'], rep_gap_ns)]
if sel.shape[0] > 0:
    for bits, grp in sel.groupby('adc_bits'):
        plt.plot(grp['frontend_BW_GHz'], grp['median_fwhm_ns'], marker='o', label=f"{bits}-bit ADC")
    plt.xlabel("Frontend bandwidth (GHz)")
    plt.ylabel("Median FWHM (ns)")
    plt.title(f"Effect of frontend bandwidth on measured FWHM (gap={rep_gap_ns} ns)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


# 3) Timing RMS vs ADC rate for representative rise & gap
plt.figure(figsize=(8,5))
sel2 = df[(np.isclose(df['rise_time_ns'], RISE_TIMES[1]*1e9)) & (np.isclose(df['gap_ns'], 2.0))]
if sel2.shape[0] > 0:
    plt.bar(sel2['adc_rate_GSPS'].astype(str), sel2['timing_error_rms_ps'])
    plt.xlabel("ADC rate (GS/s)")
    plt.ylabel("Timing error RMS (ps)")
    plt.title(f"Timing RMS vs ADC rate (rise={RISE_TIMES[1]*1e9:.2f} ns, gap=2 ns)")
    plt.tight_layout()


# 4) Example waveforms (fast vs slow front-end) for a 2 ns gap
rise_fast = RISE_TIMES[0]  # fastest tested
rise_slow = RISE_TIMES[-1]  # slowest tested
bw_fast = 0.35 / rise_fast
bw_slow = 0.35 / rise_slow
h_fast = rc_impulse_response(T - T[0], bw_fast)
h_slow = rc_impulse_response(T - T[0], bw_slow)
gap_example = 2e-9
sig_example = (gaussian_pulse(T, BASE_T0, PULSE_FWHM, PULSE_AMP)
               + gaussian_pulse(T, BASE_T0 + gap_example, PULSE_FWHM, PULSE_AMP))
sig_fast = fft_convolve(sig_example, h_fast)
sig_slow = fft_convolve(sig_example, h_slow)

# FFTs
Yf_fast = np.abs(np.fft.rfft(sig_fast, n=FFT_LEN))
Yf_slow = np.abs(np.fft.rfft(sig_slow, n=FFT_LEN))
Yf_fast_db = 20*np.log10(Yf_fast/np.max(Yf_fast) + 1e-16)
Yf_slow_db = 20*np.log10(Yf_slow/np.max(Yf_slow) + 1e-16)

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(T*1e9, sig_example); plt.title("Ideal pulses (2 ns FWHM, 2 ns gap)"); plt.xlim(40,70); plt.ylabel("Amp")
plt.subplot(3,1,2)
plt.plot(T*1e9, sig_fast, label=f"Fast BW ≈ {bw_fast/1e9:.2f} GHz"); 
plt.plot(T*1e9, sig_slow, label=f"Slow BW ≈ {bw_slow/1e9:.2f} GHz", alpha=0.7)
plt.xlim(40,70); plt.legend(); plt.ylabel("Amp"); plt.title("After frontend (rise-time limited)")
plt.subplot(3,1,3)
plt.semilogx(freqs/1e9, Yf_fast_db, label='fast'); plt.semilogx(freqs/1e9, Yf_slow_db, label='slow', alpha=0.7)
plt.xlim(1e-3, 10); plt.xlabel("Frequency (GHz)"); plt.ylabel("Relative dB"); plt.legend(); plt.title("Spectrum after frontend")
plt.tight_layout()
