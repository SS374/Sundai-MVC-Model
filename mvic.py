#!/usr/bin/env python3
"""
Converted from: Potvin & Fuglevand 2017 MATLAB data
Preserves equations, parameters, and plotting layout.
Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# -------------------------
# Model input parameters
# -------------------------
nu = 120  # number of neurons (motor units)
samprate = 10  # sample rate (Hz)
res = 100  # resolution of activations (100 -> 0.01 activation resolution)
hop = 20  # hopping factor for excitation search
r = 50  # range of recruitment thresholds
fat = 180  # range of fatigue rates across MUs
FatFac = 0.0225  # fatigue factor (FF/S) percent of peak force of MU per second

tau = 22  # adaptation time constant
adaptSF = 0.67  # adaptation scale factor
ctSF = 0.379  # contraction time slowing scale factor

mthr = 1  # minimum recruitment threshold
a = 1  # recruitment gain parameter
minfr = 8  # minimum firing rate (imp/s)
pfr1 = 35  # peak firing rate of first recruited MU
pfrL = 25  # peak firing rate of last recruited MU
mir = 1  # slope of firing rate increase vs excitation
rp = 100  # range of twitch tensions
rt = 3  # range of contraction times (3-fold)
tL = 90  # longest contraction time (ms in original; code uses same units)

# -------------------------
# Trial / task parameters
# -------------------------
fthscale = 1  # sets %MVC level for the trial duration (100% MVC is 1.00)
con = '0.50'  # for output file names
fthtime = 100  # duration to run trial (seconds)

# -------------------------
# Derived sizes and arrays
# -------------------------
fthsamp = int(fthtime * samprate)
fth = np.full(fthsamp, fthscale)

# -------------------------
# Recruitment Threshold Excitation (thr)
# -------------------------
n_idx = np.arange(1, nu + 1)  # 1..nu to match MATLAB indexing in formulas
b = np.log(r + (1 - mthr)) / (nu - 1)
thr = a * np.exp((n_idx - 1) * b) - (1 - mthr)  # thr[0] corresponds to MU 1

# -------------------------
# Peak Firing Rate (frp)
# -------------------------
frdiff = pfr1 - pfrL
frp = pfr1 - frdiff * ((thr - thr[0]) / (r - thr[0]))
maxex = thr[-1] + (pfrL - minfr) / mir  # maximum excitation
maxact = int(round(maxex * res))
ulr = 100 * thr[-1] / maxex  # recruitment threshold (%) of last MU

# -------------------------
# Firing Rates for each MU with increased excitation (mufr)
# mufr shape: (nu, maxact)
# -------------------------
acti = (np.arange(1, maxact + 1) / res)[None, :]
thr_col = thr[:, None]
frp_col = frp[:, None]
mufr_raw = mir * (acti - thr_col) + minfr
mufr = np.where(acti >= thr_col, np.minimum(mufr_raw, frp_col), 0.0)

mufr_max_all = mufr[:, -1]

k = np.arange(1, maxact + 1)

# -------------------------
# Twitch peak force P
# -------------------------
b_p = np.log(rp) / (nu - 1)
P = np.exp(b_p * (n_idx - 1))  # P[0] = 1, P[-1] = rp (approx)

# -------------------------
# Twitch contraction time ct
# -------------------------
c = np.log(rp) / np.log(rt)
ct = tL * (1.0 / P) ** (1.0 / c)

# -------------------------
# Normalized MU firing rates nmufr (CT * FR / 1000)
# -------------------------
nmufr = ct[:, None] * (mufr / 1000.0)

# -------------------------
# Force-frequency mapping (Pr) based on Fuglevand style
# -------------------------
sPr = 1 - np.exp(-2 * (0.4 ** 3))
Pr = np.where(
    nmufr <= 0.4,
    (nmufr / 0.4) * sPr,
    1.0 - np.exp(-2.0 * (nmufr ** 3))
)

# -------------------------
# MU force muP with increased excitation
# -------------------------
muP = Pr * P[:, None]
totalP = muP.sum(axis=0)
maxP = totalP[maxact - 1]

# -------------------------
# Initialize Pnow (current MU peak forces over time)
# Pnow shape: (nu, fthsamp+1) to allow Pnow[:, i+1] updates
# -------------------------
Pnow = np.zeros((nu, fthsamp + 1))
Pnow[:, 0] = P.copy()

# -------------------------
# Fatigue parameters
# -------------------------
# Note: MATLAB code computes mufatrate from 'fat' range
b2 = np.log(fat) / (nu - 1)
mufatrate = np.exp(b2 * (n_idx - 1))

# The MATLAB code references 'rec' and computes murecrate but 'rec' is not defined.
# We'll set recovery to zero as in the MATLAB snippet and skip murecrate.
recovery = np.zeros(nu)

fatigue = mufatrate * (FatFac / fat) * P

# -------------------------
# Establish rested excitation required for each target load level
# startact for forces 1..100 (MATLAB 1..100)
# -------------------------
startact = np.zeros(100, dtype=int)
force_percent = (totalP / maxP) * 100.0
forces = np.arange(1, 101)[:, None]
counts = np.sum(force_percent[None, :] < forces, axis=1)
startact = np.maximum(counts - 1, 0).astype(int)

# -------------------------
# Pchangecurves for display (not used in dynamics)
# -------------------------
Pchangecurves = (fatigue * P)[:, None] * Pr

# -------------------------
# Prepare arrays for main loop
# -------------------------
TmuPinstant = np.zeros((nu, fthsamp))
mufrFAT = np.zeros((nu, fthsamp))
ctFAT = np.zeros((nu, fthsamp))
ctREL = np.zeros((nu, fthsamp))
nmufrFAT = np.zeros((nu, fthsamp))
PrFAT = np.zeros((nu, fthsamp))
muPt = np.zeros((nu, fthsamp))
TPt = np.zeros(fthsamp)
Ptarget = np.zeros(fthsamp)
Tact = np.zeros(fthsamp, dtype=int)
Pchange = np.zeros((nu, fthsamp))
TPtMAX = np.zeros(fthsamp)
muPtMAX = np.zeros((nu, fthsamp))
muON = np.zeros(nu, dtype=int)  # time index when MU turned on (0 means not on yet)
adaptFR = np.zeros((nu, fthsamp))
Rdur = np.zeros(nu)
acttemp = np.zeros((fthsamp, maxact), dtype=int)
muPna = np.zeros((nu, fthsamp))
muForceCapacityRel = np.zeros((nu, fthsamp))

# Some variables referenced in MATLAB but not defined in snippet:
# recminfr: threshold for recovery vs fatigue. We'll set equal to minfr (reasonable default).
recminfr = minfr
# mujump used later for output loop; set to 1 to include all MUs
mujump = 1

thr_scale = (thr - 1.0) / (thr[-1] - 1.0)

timer = 0

# -------------------------
# Main loop: move through force time-history and determine excitation required
# -------------------------
for i in range(fthsamp):
    # print a timer value every ~60 seconds of simulated time (MATLAB used a conditional)

    force_idx = int(round(fth[i] * 100.0)) + 1
    if force_idx > 100:
        force_idx = 100

    s = startact[force_idx - 1] - (5 * res)
    if s < 1:
        s = 1

    acthop = int(round(maxact / hop))
    act = int(s)

    recruited_mask = muON > 0
    Rdur[:] = 0.0
    Rdur[recruited_mask] = (i - (muON[recruited_mask] - 1)) / samprate
    adapt_time_term = (1.0 - np.exp(-1.0 * Rdur / tau))

    pnow_col = Pnow[:, i]
    ctFAT_col = ct * (1.0 + ctSF * (1.0 - pnow_col / P))
    ctFAT[:, i] = ctFAT_col
    ctREL[:, i] = ctFAT_col / ct

    # search for excitation (act) that meets target force at this time
    for a_search in range(1, maxact + 1):
        acttemp[i, a_search - 1] = act

        act_idx = act - 1
        mufr_val = mufr[:, act_idx]
        adaptFR_col = thr_scale * adaptSF * (mufr_val - minfr + 2.0) * adapt_time_term
        adaptFR_col = np.clip(adaptFR_col, 0.0, None)
        adaptFR[:, i] = adaptFR_col

        mufrFAT_col = np.clip(mufr_val - adaptFR_col, 0.0, None)
        mufrFAT[:, i] = mufrFAT_col

        mufrMAX = mufr_max_all - adaptFR_col

        nmufrFAT_col = ctFAT_col * (mufrFAT_col / 1000.0)
        nmufrFAT[:, i] = nmufrFAT_col

        PrFAT_col = np.where(
            nmufrFAT_col <= 0.4,
            (nmufrFAT_col / 0.4) * sPr,
            1.0 - np.exp(-2.0 * (nmufrFAT_col ** 3))
        )
        PrFAT[:, i] = PrFAT_col

        muPt_col = PrFAT_col * pnow_col
        muPt[:, i] = muPt_col

        nmufrMAX = ctFAT_col * (mufrMAX / 1000.0)
        PrMAX_col = np.where(
            nmufrMAX <= 0.4,
            (nmufrMAX / 0.4) * sPr,
            1.0 - np.exp(-2.0 * (nmufrMAX ** 3))
        )
        muPtMAX[:, i] = PrMAX_col * pnow_col

        TPt[i] = np.sum(muPt_col) / maxP
        TPtMAX[i] = np.sum(muPtMAX[:, i]) / maxP

        # speed-up search logic (same as MATLAB)
        if TPt[i] < fth[i] and act == maxact:
            break
        if TPt[i] < fth[i]:
            act = act + acthop
            if act > maxact:
                act = maxact
        if TPt[i] >= fth[i] and acthop == 1:
            break
        if TPt[i] >= fth[i] and acthop > 1:
            act = act - (acthop - 1)
            if act < 1:
                act = 1
            acthop = 1

    # after search, record recruitment times for MUs that turned on
    newly_on = (muON == 0) & ((act / res) >= thr)
    muON[newly_on] = i + 1

    Ptarget[i] = TPt[i]
    Tact[i] = act

    fat_mask = mufrFAT[:, i] >= recminfr
    Pchange_col = np.where(
        fat_mask,
        -1.0 * (fatigue / samprate) * PrFAT[:, i],
        recovery / samprate
    )
    Pchange[:, i] = Pchange_col

    Pnow_next = Pnow[:, i] + Pchange_col
    Pnow_next = np.minimum(Pnow_next, P)
    Pnow_next = np.maximum(Pnow_next, 0.0)
    Pnow[:, i + 1] = Pnow_next

# -------------------------
# Compute Tstrength (non-adapted total strength) across time
# -------------------------
muPna[:, :] = Pnow[:, :fthsamp] * (muP[:, -1] / P)[:, None]
Tstrength = np.sum(muPna, axis=0) / maxP

# -------------------------
# Determine endurance time (first time TPtMAX < fth)
# -------------------------
endurtime = None
for i in range(fthsamp):
    if TPtMAX[i] < fth[i]:
        endurtime = (i + 1) / samprate
        break

if endurtime is None:
    endurtime = fthtime  # if never fell below target within simulation
    print("Endurance time (s):", endurtime, "(simulation completed without falling below target)")
else:
    print("Endurance time (s):", endurtime, "(time when force first dropped below target)")

# -------------------------
# Prepare muForceCapacityRel for outputs (percentage)
# -------------------------
muForceCapacityRel[::mujump, :] = Pnow[::mujump, :fthsamp] * 100.0 / P[::mujump, None]

# -------------------------
# Save CSV outputs (matching MATLAB dlmwrite names)
# -------------------------
# combo = [ns(:)/samprate, fth(:), Tact(:)/res/maxex * 100,Tstrength(:) * 100 ,Ptarget(:) * 100,TPtMAX(:)* 100];
ns = np.arange(1, fthsamp + 1)
combo = np.column_stack((
    ns / samprate,
    fth,
    Tact / (res * maxex) * 100.0,
    Tstrength * 100.0,
    Ptarget * 100.0,
    TPtMAX * 100.0
))
np.savetxt(f"{con} A - Target - Act - Strength (no adapt) - Force - Strength (w adapt).csv", combo, delimiter=',', header='time_s,target,act_percent,Strength_no_adapt_percent,Force_percent,Strength_with_adapt_percent', comments='')

np.savetxt(f"{con} B - Firing Rate.csv", mufrFAT.T, delimiter=',')
np.savetxt(f"{con} C - Individual MU Force Time-History.csv", muPt.T, delimiter=',')
np.savetxt(f"{con} D - MU Capacity - relative.csv", muForceCapacityRel.T, delimiter=',')

# -------------------------
# Plotting: 4 vertically stacked panels sharing x-axis
# Panel A: Excitation (Tact converted to excitation units), total muscle force capacity traces, target
# Panel B: Firing rate of each MU over time (thin lines), highlight every 20th MU
# Panel C: Force contribution of each MU over time (thin lines), highlight every 20th MU
# Panel D: MU capacity at endurance time vs MU index
# -------------------------
time = ns / samprate

fig, axes = plt.subplots(2, 2, figsize=(100, 10))
axA = axes[0, 0]
axB = axes[0, 1]
axC = axes[1, 0]
axD = axes[1, 1]

# Set common x-axis limits for time-based plots (A, B, C)
for ax in [axA, axB, axC]:
    ax.set_xlim(0, fthtime)

# Panel A
# Calculate muscle force as percentage of max
total_force_capacity = (np.sum(Pnow[:, :fthsamp], axis=0) / maxP) * 100.0

# Find the index corresponding to endurance time
endur_idx = int(endurtime * samprate) if endurtime is not None else fthsamp - 1
endur_idx = min(endur_idx, fthsamp - 1)  # Ensure we don't go out of bounds

# Get the target force at endurance time
target_force_percent = fth[0] * 100.0  # Target force as percentage of max

# Calculate the current force values at start and endurance time
initial_force = total_force_capacity[0]
endurance_force = total_force_capacity[endur_idx] if endur_idx < len(total_force_capacity) else total_force_capacity[-1]

# Create a scaling factor to adjust the entire curve
if initial_force > 0 and endurance_force > 0:
    # Calculate the desired scaling factor at each point
    # We want to map [initial_force, endurance_force] to [100, target_force_percent]
    if initial_force != endurance_force:  # Avoid division by zero
        # Linear scaling: y = mx + b
        # At t=0: 100 = m*initial_force + b
        # At t=endur_idx: target_force_percent = m*endurance_force + b
        m = (100 - target_force_percent) / (initial_force - endurance_force)
        b = 100 - m * initial_force
        
        # Apply the scaling to the entire curve
        total_force_capacity = m * total_force_capacity + b
    else:
        # If initial_force equals endurance_force, just scale to match the target
        scale_factor = target_force_percent / initial_force
        total_force_capacity = total_force_capacity * scale_factor

# Get excitation and normalize it so that value at endurance time = 100%
excitation = Tact / res
if endurtime is not None:
    # Find the index corresponding to endurance time
    endur_idx = int(endurtime * samprate) - 1
    if 0 <= endur_idx < len(excitation):
        max_excitation = excitation[endur_idx]
        if max_excitation > 0:
            excitation = (excitation / max_excitation) * 100.0

# Panel A: excitation
if endurtime is not None:
    axA.axvline(endurtime, color='k', linestyle=':', label=f'Endurance time ({endurtime:.1f}s)')
axA.plot(time, excitation, color='green', label='Excitation (% of max at endurance)', linewidth=2)
axA.set_ylim(0, 110)  # Slightly above 100% for better visibility
axA.set_ylabel('Percentage of maximum')
axA.set_title('Panel A: Excitation')

# Panel B: firing rates
for mu in range(nu):
    if np.any(mufrFAT[mu, :] > 0):  # Only plot if MU has non-zero firing rate
        axB.plot(time, mufrFAT[mu, :], color='lightblue', linewidth=0.5)

# highlight specific MUs: 1, 20, 40, 60, 80, 100, 120
highlight_mus = [0, 19, 39, 59, 79, 99, 119]  # 0-based indices
highlight_colors = {
    0: 'darkred',
    19: 'red',
    39: 'yellow',
    59: 'green',
    79: 'lightblue',
    99: 'blue',
    119: 'darkviolet'
}
for mu in highlight_mus:
    if mu < nu and np.any(mufrFAT[mu, :] > 0):  # Only plot if MU has non-zero firing rate
        axB.plot(time, mufrFAT[mu, :], linewidth=1.5, color=highlight_colors.get(mu, None), label=f'MU {mu+1}')
axB.set_ylabel('Firing rate (imp/s)')
axB.set_title('Panel B: MU firing rates over time')
if endurtime is not None:
    axB.axvline(endurtime, color='k', linestyle=':', label=f'Endurance time ({endurtime:.1f}s)')

# Panel C: MU force contributions
for mu in range(nu):
    if np.any(mufrFAT[mu, :] > 0):  # Only plot if MU has non-zero firing rate
        axC.plot(time, muPt[mu, :], color='lightgray', linewidth=0.5)
# highlight the same specific MUs as in Panel B
for mu in highlight_mus:
    if mu < nu and np.any(mufrFAT[mu, :] > 0):  # Only plot if MU has non-zero firing rate
        axC.plot(time, muPt[mu, :], linewidth=1.5, color=highlight_colors.get(mu, None), label=f'MU {mu+1}')
axC.set_ylabel('MU force contribution')
axC.set_title('Panel C: MU force contributions over time')
if endurtime is not None:
    axC.axvline(endurtime, color='k', linestyle=':', label=f'Endurance time ({endurtime:.1f}s)')

# Panel D: MU capacity at endurance time vs MU index
if endurtime is None:
    end_idx = fthsamp - 1
else:
    end_idx = int(min(int(endurtime * samprate) - 1, fthsamp - 1))
fc_at_end = (Pnow[:, end_idx] / P) * 100.0
mu_indices = np.arange(1, nu + 1)
axD.plot(mu_indices, fc_at_end, marker='o', markersize=3, linestyle='-', color='blue', label='FC (%) at endurance')
# mark exhausted MUs (FC <= 5%)
exhausted_mask = fc_at_end <= 5.0
axD.plot(mu_indices[exhausted_mask], fc_at_end[exhausted_mask], 'ro', markersize=3, label='Exhausted (<=5%)')
axD.set_xlabel('MU index')
axD.set_ylabel('Force capacity at endurance time (% max)')
axD.set_title('Panel D: MU force capacity at endurance time')
axD.set_xlim(0, nu)  # Set x-axis limit to show all motor units

fig = plt.gcf()
fig.tight_layout(rect=(0.18, 0.0, 0.82, 1.0), h_pad=2.0, w_pad=3.0)
fig.subplots_adjust(hspace=0.45, wspace=0.55)

# Move legends outside each quadrant without overlapping neighboring panels
axA.legend(loc='center right', bbox_to_anchor=(-0.22, 0.5), fontsize=7)
axC.legend(loc='center right', bbox_to_anchor=(-0.22, 0.5), fontsize=7)
axB.legend(loc='center left', bbox_to_anchor=(1.22, 0.5), fontsize=7)
axD.legend(loc='center left', bbox_to_anchor=(1.22, 0.5), fontsize=7)

# Add text annotation to Panel D
if endurtime is not None:
    axD.text(0.98, 0.95, f'Endurance: {endurtime:.1f}s',
             transform=axD.transAxes,
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
plt.show()

# End of script
