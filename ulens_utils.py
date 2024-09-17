import os

import astropy.units as u
import emcee
import jenkspy
import numpy as np
import scipy
import scipy.optimize as op
import toml
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time


class config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.steps = self.parameters_to_fit
        self.parameters_to_fit = list(self.parameters_to_fit.keys())
        self.parameters = self.parameters_to_fit + self.parameters_fixed
        self.n_param = len(self.parameters_to_fit)
        self.param_fix = {i: self.param_init[i] for i in self.parameters_fixed}


global lk, a1, factot, loga1


def bin(
    data: np.ndarray,
    saxis: int = 0,
    vaxis: int = 1,
    waxis: int = 2,
    nsigma: int = 3,
    nbin: int = 20,
) -> np.ndarray:
    """
    INPUT:
        data[np.ndarray]
        saxis[int]: axis of the index to be binned, should be sorted
        vaxis[int]: axis of the value to be binned
        waxis[int]: axis of the binning weight
        nsigma[int]: sigma clipping threshold, default 3
        nbin[int]: number of bins, default 20
    """
    data = np.array(sorted(data, key=lambda x: x[saxis]))
    breaks = jenkspy.jenks_breaks(data[:, saxis], int(nbin))
    output = []
    for i in range(len(breaks) - 1):
        lower = breaks[i] - (i == 0)
        upper = breaks[i + 1]
        index = np.logical_and(data[:, saxis] > lower, data[:, saxis] <= upper)
        tmp = data[index]
        mask = sigma_clip(tmp[:, vaxis], sigma=nsigma, masked=True).mask
        tmp = tmp[~mask]
        w = tmp[:, waxis] ** -2
        wtot = np.sum(w)
        line = [np.sum(w * tmp[:, k]) / wtot for k in range(len(tmp[0]))]
        line[waxis] = wtot**-0.5
        # line[waxis] = (np.sum(w * (tmp[:, vaxis] - line[vaxis]) ** 2) / wtot) ** 0.5
        if line[waxis] < 1e-8 and len(tmp) == 1:
            print(tmp, len(tmp))
            line[waxis] = tmp[0, waxis]
        output += [line]
    return np.array(output)


def loadtxt(fn: str, subs: bool = True, usecols: tuple = (0, 1, 2)) -> np.ndarray:
    if fn.endswith(".pysis5"):
        usecols = (1, -2, -1)
    if fn.endswith(".pysis"):
        usecols = (0, 2, 3)

    df = np.loadtxt(fn, usecols=usecols)
    if subs:
        if df[0][0] > 2450000:
            df[:, 0] -= 2450000
    return df


def mag2flux(data: np.ndarray, refmag: float = 18) -> np.ndarray:
    """
    INPUT:
        data[np.ndarray] : [[time1, mag1, err1], [time2, mag2, err2], ...]
        refmag[float]    : reference magnitude, default 18
    OUTPUT:
        data[np.ndarray] : [[time1, flux1, ferr1], [time2, flux2, ferr2], ...]
    """
    _data = data.copy()
    if len(_data) == 1 and _data.ndim == 1:
        return 10 ** (0.4 * (refmag - _data[0]))
    mag = _data[:, 1]
    err = _data[:, 2]
    flux = 10 ** (0.4 * (refmag - mag))
    ferr = flux * err * 0.4 * np.log(10)
    _data[:, 1] = flux
    _data[:, 2] = ferr
    return _data


def flux2mag(data: np.ndarray, refmag: float = 18) -> np.ndarray:
    """
    INPUT:
        data[np.ndarray] : [[time1, flux1, ferr1], [time2, flux2, ferr2], ...]
        refmag[float]    : reference magnitude, default 18
    OUTPUT:
        data[np.ndarray] : [[time1, mag1, err1], [time2, mag2, err2], ...]
    """
    _data = data.copy()
    refmag = 18
    if np.array(data).ndim == 1:
        return -2.5 * np.log10(data) + refmag
    flux = data[:, 1]
    ferr = data[:, 2]
    mag = -2.5 * np.log10(flux) + refmag
    magerr = ferr / flux * 2.5 / np.log(10)
    _data[:, 1] = mag
    _data[:, 2] = magerr
    return _data


def sample_gen(par1, par2, n, method="gauss"):
    if method == "gauss":
        return par1 + par2 * np.random.randn(n)
    if method == "uniform":
        return np.random.uniform(low=par1, high=par2, size=n)
    if method == "log-uniform":
        return np.exp(np.random.uniform(low=np.log(par1), high=np.log(par2), size=n))


def getchi2_single(
    data: np.ndarray, model: np.ndaray, blending: bool = False, ap: bool = False
):
    """
    INPUT:
        data[np.ndarray] :
            [[time1, time2, ...], [flux1, flux2, ...], [ferr1, ferr2, ...]]

        model[np.ndarray]: [A1, A2, A3, ...]
        blending[bool]   : blending flag
        ap[bool]         : aperture flux correction flag

    OUTPUT:
        chi2[float]      : chi2 value
        f[np.ndarray]    : [f1, f2, f3], source flux, lens flux and ap flux
    """
    dat = data.T
    y = dat[1] / dat[2]
    if ap and blending:
        A = np.vstack([model / dat[2], 1 / dat[2], dat[3] / dat[2]]).T
    elif blending:
        A = np.vstack([model / dat[2], 1 / dat[2]]).T
    elif ap:
        A = np.vstack([model / dat[2], dat[3] / dat[2]]).T
    else:
        A = np.vstack([model / dat[2]]).T
    res = np.linalg.lstsq(A, y, rcond=None)

    f = np.append(res[0], [0, 0])
    if ap and not blending:
        f[2] = f[1]
        f[1] = 0
    if len(res[1]) == 0:
        return 1e16, f[:3]
    return res[1][0], f[:3]


def VBBL2mcmc_planet(param: dict, plx=False) -> dict:
    """
    INPUT : t0 u0 tE rho s q alpha
    OUTPUT : new_t0 new_u0/w new_teff tstar logw logq alpha
    """
    keep = ["alpha"]
    if plx:
        keep += ["pi1", "pi2"]
    _param = {i: param[i] for i in keep}
    t0 = param["t0"]
    u0 = param["u0"]
    tE = param["tE"]
    rho = param["rho"]
    s = param["s"]
    q = param["q"]
    a = param["alpha"]
    qf = qfac(s)
    w = qf * q**2 / (1 + q) ** 2
    if s > 1:
        offset = q / (1 + q) * (s - 1 / s)
        # _y1 = u0*np.sin(alpha) - tau*np.cos(alpha)
        # _y2 = -u0*np.cos(alpha) - tau*np.sin(alpha)
        u0 += offset * np.sin(a)
        t0 += offset * np.cos(a) * tE
    tstar = rho * tE
    teff = abs(u0 * tE)
    _param["t0"] = t0
    _param["u0/w"] = u0 / w * q**0.5
    _param["teff"] = teff
    _param["tstar"] = tstar
    _param["logw"] = np.log10(w)
    _param["logq"] = np.log10(q)
    _param["flag_wide"] = 1 * (param["s"] > 1)
    return _param


def mcmc2VBBL_planet(param, plx=False):
    """
    INPUT : new_t0 new_u0/w new_teff tstar logw logq alpha
    OUTPUT: t0 u0 tE rhos s q alpha
    """
    keep = ["alpha"]
    if plx:
        keep += ["pi1", "pi2"]
    _param = {i: param[i] for i in keep}
    t0 = param["t0"]
    u0_w = param["u0/w"]
    teff = param["teff"]
    tstar = param["tstar"]
    w = 10 ** param["logw"]
    q = 10 ** param["logq"]
    a = param["alpha"]
    qf = w / q**2 * (1 + q) ** 2
    s = invqfac(qf)
    u0 = u0_w * w / q**0.5
    tE = abs(teff / u0)
    rho = tstar / tE
    if param["flag_wide"]:
        s = 1 / s
        offset = q / (1 + q) * (s - 1 / s)
        u0 -= offset * np.sin(a)
        t0 -= offset * np.cos(a) * tE
    _param["t0"] = t0
    _param["u0"] = u0
    _param["tE"] = tE
    _param["rho"] = rho
    _param["s"] = s
    _param["q"] = q
    return _param


def mcmc2VBBL_planet2(param, keep=["alpha", "pi1", "pi2"]):
    """
    INPUT : new_t0 new_u0/w new_teff tstar logw logq alpha
    OUTPUT: t0 u0 tE rhos s q alpha
    """
    _param = {i: param[i] for i in keep}
    t0 = param["t0"]
    u0 = param["u0"]
    teff = param["teff"]
    tstar = param["tstar"]
    s = 10 ** param["logs"]
    q = 10 ** param["logq"]
    a = param["alpha"]
    sc = (1 + q) ** 0.5
    u0 = u0 / sc
    tE = abs(teff / u0)
    rho = tstar / tE
    if s > 1:
        offset = q / (1 + q) * (s - 1 / s)
        u0 -= offset * np.sin(a)
        t0 -= offset * np.cos(a) * tE
    _param["t0"] = t0
    _param["u0"] = u0
    _param["tE"] = tE
    _param["rhos"] = rho
    _param["s"] = s
    _param["q"] = q
    if "pi1" in param and "pi2" in param:
        _param["pi1"] = param["pi1"] / sc
        _param["pi2"] = param["pi2"] / sc
    return _param


def qfac(s):
    st = s + 1 / s
    cphi = 3 / 4 * st * (1 - (1 - 32 / 9 / st**2) ** 0.5)
    sphi = (1 - cphi**2) ** 0.5
    qfac = 4 * sphi**3 / (st - 2 * cphi) ** 2
    return qfac


def invqfac(q):
    def f(x):
        return qfac(x) - q

    return op.brentq(f, 1e-5, 1)


def VBBL2mcmc_stellar(param, plx=False):
    """
    INPUT : t0 u0 tE rho s q alpha piEN piEE
    OUTPUT : new_t0 new_u0 new_teff tstar logs logq alpha piEN piEE
    """
    keep = ["alpha"]
    if plx:
        keep += ["pi1", "pi2"]
    _param = {i: param[i] for i in keep}
    t0 = param["t0"]
    u0 = param["u0"]
    tE = param["tE"]
    rhos = param["rhos"]
    s = param["s"]
    q = param["q"]
    a = param["alpha"]
    if s > 1:
        offset = q / (1 + q) * (s - 1 / s)
        # _y1 = u0*np.sin(alpha) - tau*np.cos(alpha)
        # _y2 = -u0*np.cos(alpha) - tau*np.sin(alpha)
        u0 += offset * np.sin(a)
        t0 += offset * np.cos(a) * tE
    tstar = rhos * tE
    teff = abs(u0 * tE)
    _param["t0"] = t0
    _param["u0"] = u0
    _param["teff"] = teff
    _param["tstar"] = tstar
    _param["logs"] = np.log10(s)
    _param["logq"] = np.log10(q)
    return _param


def mcmc2VBBL_stellar(param, keep=["alpha", "pi1", "pi2"]):
    """
    INPUT : new_t0 new_u0 new_teff tstar logs logq alpha piEN piEE
    OUTPUT: t0 u0 tE rhos s q alpha piEN piEE
    """
    _param = {i: param[i] for i in keep}
    t0 = param["t0"]
    u0 = param["u0"]
    teff = param["teff"]
    s = 10 ** param["logs"]
    q = 10 ** param["logq"]
    a = param["alpha"]
    tE = abs(teff / u0)
    if not "rhos" in param:
        tstar = param["tstar"]
        rhos = tstar / tE
    else:
        rhos = param["rhos"]
    if s > 1:
        offset = q / (1 + q) * (s - 1 / s)
        u0 -= offset * np.sin(a)
        t0 -= offset * np.cos(a) * tE
    _param["t0"] = t0
    _param["u0"] = u0
    _param["tE"] = tE
    _param["rhos"] = rhos
    _param["s"] = s
    _param["q"] = q
    return _param


def c2w(param: dict) -> dict:
    """
    close (s, q) to wide (s, q) parameter conversion
    """
    qc = param["q"]
    sc = param["s"]
    qw = qc * (1 - qc) ** -2
    sw = (sc**-1) * (1 + qc) * (qc**2 - qc + 1) ** -0.5
    _param = param.copy()
    _param["s"] = sw
    _param["q"] = qw
    return _param


def dw(q):
    return ((1 + q ** (1 / 3)) ** 3 / (1 + q)) ** 0.5


def radec(ra_dec: str) -> tuple[float, float]:
    """
    radec[str] to deg
    """
    ra, dec = ra_dec.split()
    sgn = 1
    if "-" in dec:
        sgn = -1
    dec = dec.replace("-", "")
    a, b, c = [float(i) for i in ra.split(":")]
    d, e, f = [float(i) for i in dec.split(":")]

    alpha = (a + b / 60 + c / 3600) * 15
    delta = sgn * (d + e / 60 + f / 3600)
    return alpha, delta


def pspl2vlti(eta: float, psi: float, thetaE: float, fbfs: float = 0) -> dict:
    """
    INPUT:
            eta   : flux ratio of the 2 images;
            psi   : position angle of the major image to minor image;
            thetaE: Einstein radius in milliarcsecond;
            fbfs  : flux ratio of the blended flux to the flux of the source;
    OUTPUT:
            The dictionary of components of the model centered on the lens;
            L -> lens; M -> major image; m -> minor image; ud -> uniform disk; x,y -> position of the source;
    """
    # NOTE: psi measured from north to east(major image)
    psi *= np.pi / 180
    if eta == 1:
        Q = 1
    else:
        A = (1 + eta) / (1 - eta)
        Q = (1 - A**-2) ** -0.5
    u = ((Q - 1) * 2) ** 0.5
    up, un = (u + (u**2 + 4) ** 0.5) / 2, (u - (u**2 + 4) ** 0.5) / 2
    # up, un = 1, -1
    Mx = np.sin(psi) * up * thetaE
    My = np.cos(psi) * up * thetaE
    mx = np.sin(psi) * un * thetaE
    my = np.cos(psi) * un * thetaE
    param = {
        "M,ud": 0.01,
        "m,ud": 0.01,
        "M,x": Mx,
        "M,y": My,
        "m,x": mx,
        "m,y": my,
        "M,f": 1 / (1 + eta),
        "m,f": eta / (1 + eta),
    }
    param["L,ud"] = 0.01
    param["L,x"] = 0.0
    param["L,y"] = 0.0
    param["L,f"] = fbfs / A
    return param


def VIS(uv: np.ndarray, wl: np.ndarray, par: dict) -> complex:
    """
    INPUT:
        uv[np.ndarray] : [u, v] coordinates in [m]
        wl[np.ndarray] : wavelength in [um]
        par[dict]      : dictionary of components, e.g. A,x:... A,y:... A,f
    OUTPUT:
        vis[complex]   : complex visibility
    """
    u, v = uv
    component = list(set([i.split(",")[0] for i in par.keys()]))
    fac = np.pi / 180 / 3600 / 1000 / 1e-6  # mas*m.um -> radians

    vis = 0
    for c in component:
        f, x, y = par[c + ",f"], par[c + ",x"] - par["M,x"], par[c + ",y"] - par["M,y"]
        vis += f * np.exp(-2j * np.pi * (u * x + v * y) * fac / wl)
    return vis


def T3(uv1: np.ndarray, uv2: np.ndarray, wl: np.ndarray, par: dict) -> complex:
    """
    INPUT:
        uv1,2[np.ndarray] : [u, v] coordinates in [m]
        wl[np.ndarray]    : wavelength in [um]
        par[dict]         : dictionary of components, e.g. A,x:... A,y:... A,f
    OUTPUT:
        t3[complex]       : bispectrum[complex]
    """
    uv3 = (-uv1[0] - uv2[0], -uv1[1] - uv2[1])
    return VIS(uv1, wl, par) * VIS(uv2, wl, par) * VIS(uv3, wl, par)


def gairmass(jd, radec_target, radec_site):
    target = SkyCoord(ra=151.7690, dec=-66.1809, unit="deg")
    bear_mountain = EarthLocation(lat=-30.47 * u.deg, lon=-70.765 * u.deg, height=0)
    obstime = Time(jd, format="jd")
    altaz = target.transform_to(AltAz(obstime=obstime, location=bear_mountain))
    return altaz.secz


def V2loop_bucket(uv, linkshead, val, countsum):
    f1s = np.dot(linkshead, uv)
    f2s = lk[val] * uv[0]
    ff1s = np.exp(np.outer(f1s, loga1))
    ff1s = np.array(
        [
            np.sum(ff1s[countsum[i] : countsum[i + 1]], axis=0)
            for i in range(len(countsum) - 1)
        ]
    )
    x = np.exp(f2s)
    vis = (ff1s * (1 - x)).sum(axis=0)
    vis /= 1 - np.exp(lk[1] * uv[0])
    return vis


# @profile
def V2loop(uv, linkshead, linksnum):
    f1s = np.dot(linkshead, uv)
    f2s = lk[linksnum] * uv[0]
    ff1s = np.exp(np.outer(f1s, loga1))
    x = np.exp(f2s)
    vis = (ff1s * (1 - x)).sum(axis=0)
    vis /= 1 - np.exp(lk[1] * uv[0])
    return vis


def T3loop(uv1, uv2, linkshead, linksnum):
    uv3 = (-uv1[0] - uv2[0], -uv1[1] - uv2[1])
    return (
        V2loop(uv1, linkshead, linksnum)
        * V2loop(uv2, linkshead, linksnum)
        * V2loop(uv3, linkshead, linksnum)
    )


def resolution_setup(step, wl, maxlen=2048):
    global lk, factot, loga1

    lk = np.zeros((maxlen, len(wl)), dtype=complex)
    fac = np.pi / 180 / 3600 / 1000 / 1e-6  # mas*m.um -> radians
    factot = fac / wl * np.pi
    # a1 = np.exp(-2j * factot)
    loga1 = -2j * factot
    for i in range(maxlen):
        lk[i] = -2j * factot * i * step


def segs2images(segs, tol=1e-8):
    segs2connect = []
    outimages = []
    for i in segs[:]:
        x = i.T
        r = x[0] ** 2 + x[1] ** 2
        x = x.T[r > 0]
        if len(x) <= 1:
            continue
        if np.sum((x[0] - x[-1]) ** 2) < tol**2:
            outimages.append(x)
        else:
            segs2connect.append(x)
    # print(len(outimages))
    # print(len(segs2connect))
    chains = connect_segs(segs2connect)
    # print()
    # print(len(outimages))
    # print(len(segs2connect))
    # print(chains)
    # print()
    for c in chains:
        img = []
        for id in c:
            img += [segs2connect[id]]
        img += [img[0][0]]
        outimages.append(np.vstack(img))
    return outimages


def connect_segs(segs, tol=0.1):
    ll = len(segs)
    if not ll:
        return []
    start = np.array([segs[i][0] for i in range(ll)])
    ends = np.array([segs[i][-1] for i in range(ll)])
    left = list(range(ll))
    now = left[0]
    chain = []
    chains = []
    while len(left) > 0:
        dist = scipy.spatial.distance.cdist([ends[now]], start[left])[0]
        next = np.argmin(dist)
        now = left[next]
        if dist[next] > tol and len(chain) > 1:
            chains += [chain]
            chain = []
        chain.append(now)
        left.remove(now)
    chains.append(chain)
    return chains


def t3phisum(t3phi: np.ndarray) -> float:
    """
    INPUT:
        t3phi[np.ndarray] : list of t3phi in [degree]

    OUTPUT:
        binned t3phi in [degree]
    """
    return np.angle(np.sum(np.exp(1j * t3phi / 180 * np.pi)), deg=True)


def unity_bin(wl, t3phi, nperbin=23):
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


def loadvlti(fn, fibre="SC", nperbin=1, errnorm=1, minerr=0.5):
    # OUTPUT:
    # T3PHI[dict], V2[dict], FLUX[dict]
    hdu = fits.open(fn)
    tel_map = dict(zip(hdu[1].data["STA_INDEX"], hdu[1].data["STA_NAME"]))

    if fibre == "SC":
        v2id = 10
        t3id = 11
        flid = 12
        wlid = 3
    elif fibre == "FT":
        v2id = 6
        t3id = 7
        flid = 8
        wlid = 4

    WL = hdu[wlid].data["EFF_WAVE"] * 1e6

    t3kwd = [
        "T3PHI",
        "T3PHIERR",
        "U1COORD",
        "V1COORD",
        "U2COORD",
        "V2COORD",
        "MJD",
        "FLAG",
    ]
    triangles = [
        tel_map[i[0]] + tel_map[i[1]] + tel_map[i[2]]
        for i in hdu[t3id].data["STA_INDEX"]
    ]
    v2kwd = ["VIS2DATA", "VIS2ERR", "UCOORD", "VCOORD", "MJD", "FLAG"]
    baselines = [tel_map[i[0]] + tel_map[i[1]] for i in hdu[v2id].data["STA_INDEX"]]

    flkwd = ["MJD", "FLUX", "FLUXERR", "FLAG"]
    tels = [tel_map[i] for i in hdu[flid].data["STA_INDEX"]]

    t3 = {}
    v2 = {}
    fl = {}
    for i, tri in enumerate(triangles):
        if tri not in t3:
            t3[tri] = []
        temp = {}
        for kwd in t3kwd:
            temp[kwd] = hdu[t3id].data[kwd][i]
        temp["WL"] = WL
        if nperbin > 1:
            wl, t3phi = temp["WL"], temp["T3PHI"]
            wl_bin, t3phi_bin, t3phierr_bin = unity_bin(wl, t3phi, nperbin)
            temp["WL"] = wl_bin
            temp["T3PHI"] = t3phi_bin
            temp["T3PHIERR"] = t3phierr_bin
        e = np.array([max(i, minerr) for i in temp["T3PHIERR"]])
        temp["T3PHIERR"] = e
        temp["T3PHIERR"] *= errnorm
        t3[tri].append(temp)

    for i, bl in enumerate(baselines):
        if bl not in v2:
            v2[bl] = []
        temp = {}
        for kwd in v2kwd:
            temp[kwd] = hdu[v2id].data[kwd][i]
        temp["WL"] = WL
        v2[bl].append(temp)

    for i, tel in enumerate(tels):
        if tel not in fl:
            fl[tel] = []
        temp = {}
        for kwd in flkwd:
            temp[kwd] = hdu[flid].data[kwd][i]
        temp["WL"] = WL
        fl[tel].append(temp)

    return t3, v2, fl


def gengrids(lists):
    mesh = np.meshgrid(*lists)
    grids = np.vstack(list(map(np.ravel, mesh))).T
    return grids


def blind_angle(uv1, uv2, pj=[1, 1]):
    fac = np.pi / 180 / 3600 / 1000 / 1e-6 / 2.2 * 2

    # NOTE: solve for x,y such that
    # u1*x + v1*y = pj[0] / fac
    # u2*x + v2*y = pj[0] / fac
    A = np.array([uv1, uv2]) * fac
    B = np.array(pj)
    x = np.linalg.solve(A, B)
    return (90 - np.angle(x[0] + 1j * x[1], deg=True)) % 180, np.dot(x, x) ** 0.5


def gamma2a1(gamma_lld):
    return 3 * gamma_lld / (2 + gamma_lld)


def t3chi2(t3data, t3model, tri=["U4U3U2", "U4U3U1", "U4U2U1", "U3U2U1"]):
    chi2 = 0
    for t in tri:
        e = t3data[t][0]["T3PHIERR"]
        dt3phi = t3data[t][0]["T3PHI"] - t3model[t][0]
        dt3phi = (dt3phi + 180) % 360 - 180
        chi2 += np.sum((dt3phi) ** 2 / e**2)
    return chi2


def loadconf(fn) -> tuple[list[dict], dict]:
    """
    INPUT  : configuration file in TOML format
    OUTPUT : list of photometry data in dictionary and configuration in dictionary
    """
    conf = toml.load(fn)
    data_path = conf["data_path"]
    phots = conf["phot"]

    for i, pho in enumerate(phots):
        eadd, escale = pho["eadd"], pho["escale"]
        dat = loadtxt(os.path.join(data_path, pho["name"]))
        dat[:, 2] = (dat[:, 2] ** 2 + eadd**2) ** 0.5
        dat[:, 2] = dat[:, 2] * escale
        pho["raw_data"] = dat
        if "mask" in pho:
            pho["mask_data"] = dat[pho["mask"]]
            dat = np.delete(dat, pho["mask"], axis=0)
        pho["data"] = dat
        if "a1" in pho and not "gamma_lld" in pho:
            pho["a1"] = gamma2a1(pho["gamma_lld"])
        pho["a1"] = 0
        pho["flux"] = mag2flux(dat)

    return phots, conf


def gsa_error(mag):
    """
    Gaia Science Alerts error model
    see Hodgkin 2021 et al.
    """
    return (
        3.43779
        - (mag / 1.13759)
        + (mag / 3.44123) ** 2
        - (mag / 6.51996) ** 3
        + (mag / 11.45922) ** 4
    )
