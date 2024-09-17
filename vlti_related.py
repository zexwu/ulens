import numpy as np
def unity_bin_bootstrap(wl, t3phi, nperbin=23):
    N = len(wl)
    bins = N // nperbin
    temp = []
    for i in range(bins):
        end = (i + 1) * nperbin
        _wl = wl[i * nperbin : end]
        _t3phi = t3phi[i * nperbin : end]
        ll = min(end, N) - i * nperbin
        _wl = np.sum(_wl**2 / (ll)) ** 0.5
        t3phi_jackknife = []
        for j in range(ll):
            _t3phi_sub = np.delete(_t3phi, j)
            t3phi_jackknife.append(t3phisum(_t3phi_sub))
        temp += [
            [_wl, np.mean(t3phi_jackknife), np.std(t3phi_jackknife) * (ll - 1) ** 0.5]
        ]
    temp = np.array(temp)
    return temp[:, 0], temp[:, 1], temp[:, 2]

