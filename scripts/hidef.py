import astropy.io.fits as pf
import matplotlib.pyplot as plt
import numpy as np

template="c:/users/duckr/Downloads/MOV_{}_SCI_IFU_PAPER_READY.fits"

bodies = {"Saturn": ['green', 1],
          "Neptune": ['red', 1],
          "Titan": ['gold', 1],
          "Uranus": ['blue', 1]}

f, ax = plt.subplots(1, figsize=(20, 6))

fmin = 1
fmax = 0


for body, (color, order) in bodies.items():

    data = pf.open(template.format(body))

    wav = data[0].data
    flux_all = data[1].data * order
    masks = data[4].data+data[5].data
    masks = masks[:-2]

    mask_c = masks==2
    mask_w = masks==1
    mask_t = masks==0
    mask_w[flux_all<0] = True
    mask_c[flux_all<0] = False

    flux_c = flux_all.copy()
    flux_c[~mask_c] = np.nan
    flux_w = flux_all.copy()
    flux_w[~mask_w] = np.nan
    flux_t = flux_all.copy()
    flux_t[~mask_t] = np.nan

    fn = flux_c[mask_c].min()
    fx = flux_c[mask_c].max()
    fmin = fn if fn < fmin else fmin
    fmax = fx if fx > fmax else fmax

    ax.plot(wav, flux_c, color=color, label=body)
    ax.plot(wav, flux_w, color=color, alpha=0.5)
    ax.plot(wav, flux_t, color=color, alpha=0.1)

gap = 0.1
df = fmax/fmin
fmax *= 1+gap
fmin *= 1-gap

ax.set_ylim(fmin, fmax)
ax.set_xlim(0.33, 2.3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()

plt.show()

