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
from scipy.stats import bootstrap


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
    Returns the binned data weighted by the inverse square of the error.

    Parameters:
        data:np.ndarray
        saxis:int axis of the index to be binned, should be sorted
        vaxis:int axis of the value to be binned
        waxis:int axis of the binning weight
        nsigma:int sigma clipping threshold, default 3
        nbin:int number of bins, default 20

    Returns:
        data_binned:np.ndarray
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
    Convert magnitude to flux

    Parameters:
        data:np.ndarray     [[time1, mag1, err1], [time2, mag2, err2], ...]
        refmag:float        reference magnitude, default 18

    Returns:
        data:np.ndarray     [[time1, flux1, ferr1], [time2, flux2, ferr2], ...]
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
    Convert flux to magnitude

    Parameters:
        data:np.ndarray     [[time1, flux1, ferr1], [time2, flux2, ferr2], ...]
        refmag:float        reference magnitude, default 18

    Returns:
        data:np.ndarray     [[time1, mag1, err1], [time2, mag2, err2], ...]
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
    data: np.ndarray, model: np.ndarray, blending: bool = False, ap: bool = False
):
    """
    Calculate chi2 for using data & model

    Parameters:
        data:np.ndarray     [[time1, time2, ...], [flux1, flux2, ...], [ferr1, ferr2, ...]]

        model:np.ndarray    [A1, A2, A3, ...]
        blending:bool       blending flag
        ap:bool             aperture correction flag

    Returns:
        chi2:float
        f:np.array          [f1, f2, f3], source flux, lens flux and ap flux
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


def espl2vlti(
    u: float, rhos: float, psi: float, thetaE: float, fbfs: float = 0
) -> list[np.ndarray]:

    psi *= np.pi / 180
    theta = np.linspace(0, 2 * np.pi, 361)[:-1]

    _srcx = u * np.sin(psi) + rhos * np.sin(theta)
    _srcy = u * np.cos(psi) + rhos * np.cos(theta)
    d = (_srcy**2 + _srcx**2) ** 0.5
    up = (d + (d**2 + 4) ** 0.5) / 2
    un = (d - (d**2 + 4) ** 0.5) / 2

    _lensx = 0
    _lensy = 0
    img1_x = _lensx + (_srcx - _lensx) / d * up
    img1_y = _lensy + (_srcy - _lensy) / d * up

    img2_x = _lensx + (_srcx - _lensx) / d * un
    img2_y = _lensy + (_srcy - _lensy) / d * un

    return [np.c_[img1_x, img1_y] * thetaE, np.c_[img2_x, img2_y][::-1] * thetaE]


def pspl2vlti(par: dict) -> dict:
    """
    Parameters[dict]:
        eta[float]      flux ratio of the 2 images;
        psi[float]      position angle of the major image to minor image;
        thetaE[float]   Einstein radius in milliarcsecond;
        fbfs[float]     flux ratio of the blended flux to the flux of the source;

    Returns[dict]:
        The dictionary of components of the model centered on the lens;
        L -> lens; M -> major image; m -> minor image; ud -> uniform disk; x,y -> position of the source;
    """
    # NOTE: psi measured from north to east(major image)
    eta, psi, thetaE, fbfs = par["eta"], par["psi"], par["thetaE"], par["fbfs"]
    psi *= np.pi / 180
    if eta == 1:
        Q = 1
    else:
        A = (1 + eta) / (1 - eta)
        Q = (1 - A**-2) ** -0.5
    u = ((Q - 1) * 2) ** 0.5
    up, un = (u + (u**2 + 4) ** 0.5) / 2, (u - (u**2 + 4) ** 0.5) / 2

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
    Compute the visibility at *uv* coordinate and *wl* with given *par* components

    Parameters:
        uv:np.ndarray       [u, v] coordinates in [m]
        wl:np.ndarray       wavelength in [um]
        par:dict            dictionary of components, e.g. A,x:... A,y:... A,f

    Returns:
        vis:complex         complex visibility
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
    Parameters:
        uv1:np.ndarray      [u, v] coordinates in [m]
        uv2:np.ndarray      [u, v] coordinates in [m]
        wl:np.ndarray       wavelength in [um]
        par:dict            dictionary of components, e.g. A,x:... A,y:... A,f

    Returns:
        t3:complex          the bispectrum[complex]
    """
    uv3 = (-uv1[0] - uv2[0], -uv1[1] - uv2[1])
    return VIS(uv1, wl, par) * VIS(uv2, wl, par) * VIS(uv3, wl, par)


def gairmass(jd, radec_target, radec_site):
    target = SkyCoord(ra=151.7690, dec=-66.1809, unit="deg")
    bear_mountain = EarthLocation(lat=-30.47 * u.deg, lon=-70.765 * u.deg, height=0)
    obstime = Time(jd, format="jd")
    altaz = target.transform_to(AltAz(obstime=obstime, location=bear_mountain))
    return altaz.secz


def lenseq(z, z1, z2, m1, m2):
    zeta_c = z.conjugate() + m1 / (z1 - z) + m2 / (z2 - z)
    return zeta_c.conjugate()


def V2direc(uv, linkshead, linksnum, xc, yc, s, q, rhos, ct):
    nn = len(lk.T)
    V = np.zeros(len(lk.T), dtype=complex)
    f1s = np.dot(linkshead, uv)
    ff1s = np.exp(np.outer(f1s, loga1))
    vec = np.array([1, 0])

    z1 = -s * q / (1 + q)
    z2 = s * 1 / (1 + q)
    m1 = 1 / (1 + q)
    m2 = 1 - m1
    z0 = xc + 1j * yc
    xys = []
    for i in range(len(linkshead)):
        _V = np.zeros(nn, dtype=complex)
        _ff1s = ff1s[i]
        vv = np.arange(linksnum[i])
        _V = np.exp(lk[vv] * uv[0])
        z = (linkshead[i][0] + ct[0] + vv * 0.001) + 1j * (linkshead[i][1] + ct[1])
        z /= (1 + q) ** 0.5
        zc = lenseq(z, z1, z2, m1, m2)
        w = np.abs(zc - z0) / rhos
        w[w > 1] = 1
        # _V += ld * np.exp(lk[j] * uv[0])
        # _V *= (1 - 0.288 * (1 - 1.5 * (1 - w**2) ** 0.5))[:, None]
        V += _V.sum(axis=0) * _ff1s
    return V, xys


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


def segs2images2(segs, tol=1e-8):
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
    chains = connect_segs2(segs2connect)
    for img in chains:
        img += [img[0][0]]
        outimages.append(np.vstack(img))
    return outimages

def connect_segs2(_segs, tol=0.1):
    segs = _segs.copy()
    ll = len(segs)
    if not ll:
        return []

    start = np.array([segs[i][0] for i in range(ll)])
    ends = np.array([segs[i][-1] for i in range(ll)])
    while len(segs) > 0:
        dist = scipy.spatial.distance.cdist(ends, start)
        idx = np.argmin(dist)
        if dist[idx] > tol:
            break
        segs[idx[0]] = np.vstack([segs[idx[0]], segs[idx[1]]])
        del segs[idx[1]]
        start = np.array([segs[i][0] for i in range(ll)])
        ends = np.array([segs[i][-1] for i in range(ll)])
    return segs


def t3phisum(t3phi: np.ndarray) -> float:
    """
    Unity binning of closure phases

    Parameters:
        t3phi:np.ndarray    list of t3phi in [degree]

    Returns:
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


def unity_bin_bootsrap(wl: np.ndarray, t3phi: np.ndarray, nperbin: int = 23):
    N = len(wl)
    bins = N // nperbin
    temp = []
    for i in range(bins):
        end = (i + 1) * nperbin
        _wl = wl[i * nperbin : end]
        _t3phi = t3phi[i * nperbin : end]
        ll = min(end, N) - i * nperbin
        # mask = sigma_clip(_t3phi, sigma=3, masked=True).mask
        # _wl = _wl[~mask]
        # _t3phi = _t3phi[~mask]

        res_t3phi = bootstrap(
            (_t3phi,),
            statistic=t3phisum,
            random_state=np.random.default_rng(),
            n_resamples=1000,
        )
        __wl = np.sum(_wl**2 / (ll)) ** 0.5
        temp += [
            [__wl, np.mean(res_t3phi.bootstrap_distribution), res_t3phi.standard_error]
        ]
    temp = np.array(temp)
    return temp[:, 0], temp[:, 1], temp[:, 2]


def loadvlti(
    fn: str,
    fibre: str = "SC",
    nperbin: int = 1,
    errnorm: float = 1.0,
    erradd: float = 0.0,
    minerr: float = 0.0,
) -> tuple[dict, dict, dict]:
    """
    Parameters:
        fn:str          filename of the fits file
        fibre:str       target type, either SC, FT or aspro2
        nperbin:int     number of data points per bin
        errnorm:float   normalization factor for error
        erradd:float    additional quadratic error
        minerr:float    minimum error threshold

    Returns:
        T3PHI[triangle][frame][WL, T3PHI, T3PHIERR, U1COORD, V1COORD, U2COORD, V2COORD, MJD, FLAG]
        V2[baseline][frame][VIS2DATA, VIS2ERR, UCOORD, VCOORD, MJD, FLAG]
        FLUX[telescope][frame][MJD, FLUX, FLUXERR, FLAG]
    """
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
    elif fibre == "aspro2":
        v2id = 5
        t3id = 6
        flid = 7
        wlid = 3

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
    if fibre == "aspro2":
        flkwd[1] = "FLUXDATA"
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
            # wl_bin, t3phi_bin, t3phierr_bin = unity_bin(wl, t3phi, nperbin)
            wl_bin, t3phi_bin, t3phierr_bin = unity_bin_bootsrap(wl, t3phi, nperbin)
            temp["WL"] = wl_bin
            temp["T3PHI"] = t3phi_bin
            temp["T3PHIERR"] = t3phierr_bin
        e = np.array([max(i, minerr) for i in temp["T3PHIERR"]])
        temp["T3PHIERR"] = e
        temp["T3PHIERR"] = (temp["T3PHIERR"] ** 2 + erradd**2) ** 0.5
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


def check_chain(fn):
    reader = emcee.backends.HDFBackend(fn)
    chain = reader.get_chain(flat=True)
    log_prob = reader.get_log_prob(flat=True)
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    print(chain[-5:])
    tau = reader.get_autocorr_time(quiet=True)
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))


def gamma2a1(gamma_lld):
    return 3 * gamma_lld / (2 + gamma_lld)


def t3chi2(
    t3data,
    t3model,
    tri=["U4U3U2", "U4U3U1", "U4U2U1", "U3U2U1"],
    sigma: float = 0,
    clip: bool = False,
) -> float:
    chi2 = 0
    for t in tri:
        e2 = t3data[t][0]["T3PHIERR"] ** 2 + sigma**2
        if sigma > 0.1:
            _chi2 = (t3data[t][0]["T3PHI"] - t3model[t][0]) ** 2 / e2
            if clip:
                _chi2 = _chi2[1:-1]
                e2 = e2[1:-1]
            chi2 += np.sum(_chi2) + np.sum(np.log(e2))
        else:
            _chi2 = (t3data[t][0]["T3PHI"] - t3model[t][0]) ** 2 / e2
            if clip:
                _chi2 = _chi2[1:-1]
            chi2 += np.sum(_chi2)
    return chi2


def loadconf(
    fn: str, mask: callable = lambda x: np.zeros(len(x[:, 0]), dtype=bool)
) -> tuple[list[dict], dict]:
    """
    Load photometry data from configuration file

    Parameters: 
        fn:str          configuration file in TOML format
        mask:callable   function return the mask to the data

    Returns: 
        list of photometry data in dictionary and configuration in dictionary
    """
    conf = toml.load(fn)
    data_path = conf["data_path"]
    phots = conf["phot"]

    for _, pho in enumerate(phots):
        eadd, escale = pho["eadd"], pho["escale"]
        dat = loadtxt(os.path.join(data_path, pho["name"]))
        dat[:, 2] = (dat[:, 2] ** 2 + eadd**2) ** 0.5
        dat[:, 2] = dat[:, 2] * escale
        pho["raw_data"] = dat
        _mask = mask(dat)
        if "mask" in pho:
            _mask[pho["mask"]] = True
        pho["mask_data"] = dat[_mask]
        pho["mask"] = np.where(_mask)[0]
        # dat = np.delete(dat, _mask, axis=0)
        dat = dat[~_mask]
        pho["data"] = dat
        if (not "a1" in pho) and "gamma_lld" in pho:
            pho["a1"] = gamma2a1(pho["gamma_lld"])
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


def pair(
    t1: np.array, t2: np.array, tol: float = 0.01
) -> tuple[np.array, np.array, np.array]:
    """
    Parameters:
        t1:np.ndarray       time series 1 to be paired
        t2:np.ndarray       time series 2 to be paired
        tol:float           tolerance for pairing

    Returns: 
        the indices of the pairs w.r.t. t1 and t2.
    """
    ind1 = []
    ind2 = []
    diff = []

    i1 = 0
    i2 = 0

    while i1 < len(t1) and i2 < len(t2):
        if t2[i2] - t1[i1] > tol:
            i1 += 1
        elif t1[i1] - t2[i2] > tol:
            i2 += 1
        else:
            ind1.append(i1)
            ind2.append(i2)
            diff.append(t1[i1] - t2[i2])
            i1 += 1
    return np.array(ind1), np.array(ind2), np.array(diff)


def cross(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def fourier(u, vertices, wl=2.2):
    fac = np.pi / 180 / 3600 / 1000 / 1e-6  # mas*m.um -> radians
    r = vertices.T * (fac / wl)
    if np.dot(u, u) < 1e-8:
        # x, y = vertices.T * np.pi / 180 / 3600 / 1000
        # area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # return area
        u = [1e-3, 1e-3]

    r1 = np.roll(r, -1, axis=1)
    ravg = (r + r1) / 2
    rdif = (r1 - r) / 2
    udravg = np.dot(u, ravg)
    udrdif = np.dot(u, rdif)
    ucrdif = cross(u, rdif)
    tmp = -2j * np.exp(-2j * np.pi * udravg) * ucrdif
    fac = np.sin(udrdif) / udrdif

    fac[np.where(abs(udrdif) < 1e-8)] = 1
    return np.sum(fac * tmp) * (4 * np.pi**2 * np.dot(u, u)) ** -1


def beta(param: dict, piS: float) -> float:
    # INPUT: VBBL parameters
    kappa = 8.144
    thetaE = 0.75
    if not "piE" in param:
        pi1, pi2 = param["pi1"], param["pi2"]
        piE2 = pi1**2 + pi2**2
        piE = piE2**0.5
    else:
        piE = param["piE"]
    if param["s"] > 1:
        sc = (param["q"] + 1) ** 0.5
        thetaE *= sc
        piE /= sc
    gamma2 = (param["dsdt"] / param["s"]) ** 2 + (param["dalphadt"]) ** 2
    beta = (
        kappa
        * (365**2)
        / 8
        / np.pi**2
        * piE
        / thetaE
        * gamma2
        * (param["s"] / (piE + piS / thetaE)) ** 3
    )
    return beta


vlti_latitude = -24.62743941  # degrees
vlti_longitude = -70.40498688  # degrees
layout_orientation = -18.984  # degrees
layout = {
    "A0": (-32.0010, -48.0130, -14.6416, -55.8116, 129.8495),
    "A1": (-32.0010, -64.0210, -9.4342, -70.9489, 150.8475),
    "B0": (-23.9910, -48.0190, -7.0653, -53.2116, 126.8355),
    "B1": (-23.9910, -64.0110, -1.8631, -68.3338, 142.8275),
    "B2": (-23.9910, -72.0110, 0.7394, -75.8987, 150.8275),
    "B3": (-23.9910, -80.0290, 3.3476, -83.4805, 158.8455),
    "B4": (-23.9910, -88.0130, 5.9449, -91.0303, 166.8295),
    "B5": (-23.9910, -96.0120, 8.5470, -98.5942, 174.8285),
    "C0": (-16.0020, -48.0130, 0.4872, -50.6071, 118.8405),
    "C1": (-16.0020, -64.0110, 5.6914, -65.7349, 134.8385),
    "C2": (-16.0020, -72.0190, 8.2964, -73.3074, 142.8465),
    "C3": (-16.0020, -80.0100, 10.8959, -80.8637, 150.8375),
    "D0": (0.0100, -48.0120, 15.6280, -45.3973, 97.8375),
    "D1": (0.0100, -80.0150, 26.0387, -75.6597, 134.8305),
    "D2": (0.0100, -96.0120, 31.2426, -90.7866, 150.8275),
    "E0": (16.0110, -48.0160, 30.7600, -40.1959, 81.8405),
    "G0": (32.0170, -48.0172, 45.8958, -34.9903, 65.8357),
    "G1": (32.0200, -112.0100, 66.7157, -95.5015, 129.8255),
    "G2": (31.9950, -24.0030, 38.0630, -12.2894, 73.0153),
    "H0": (64.0150, -48.0070, 76.1501, -24.5715, 58.1953),
    "I1": (72.0010, -87.9970, 96.7106, -59.7886, 111.1613),
    "I2": (80, -24, 83.456, 3.330, 90),  # -- XY are correct, A0 is guessed!
    "J1": (88.0160, -71.9920, 106.6481, -39.4443, 111.1713),
    "J2": (88.0160, -96.0050, 114.4596, -62.1513, 135.1843),
    "J3": (88.0160, 7.9960, 80.6276, 36.1931, 124.4875),
    "J4": (88.0160, 23.9930, 75.4237, 51.3200, 140.4845),
    "J5": (88.0160, 47.9870, 67.6184, 74.0089, 164.4785),
    "J6": (88.0160, 71.9900, 59.8101, 96.7064, 188.4815),
    "K0": (96.0020, -48.0060, 106.3969, -14.1651, 90.1813),
    "L0": (104.0210, -47.9980, 113.9772, -11.5489, 103.1823),
    "M0": (112.0130, -48.0000, 121.5351, -8.9510, 111.1763),
    "U1": (-16.0000, -16.0000, -9.9249, -20.3346, 189.0572),
    "U2": (24.0000, 24.0000, 14.8873, 30.5019, 190.5572),
    "U3": (64.0000, 48.0000, 44.9044, 66.2087, 199.7447),
    "U4": (112.0000, 8.0000, 103.3058, 43.9989, 209.2302),
    #'N0':[188,     -48,     0,        0, 188],
    #'J7':[ 88,    -167,     0,        0, 200]
}


def get_uv(T1, T2, ha, radec):
    b = [
        layout[T1][2] - layout[T2][2],
        layout[T1][3] - layout[T2][3],
        0.0,
    ]  # assumes you *DO NOT* combine ATs and UTs

    # -- projected baseline
    ch_ = np.cos(ha * np.pi / 180.0)
    sh_ = np.sin(ha * np.pi / 180.0)
    cl_ = np.cos(vlti_latitude * np.pi / 180.0)
    sl_ = np.sin(vlti_latitude * np.pi / 180.0)
    cd_ = np.cos(radec[1] * np.pi / 180.0)
    sd_ = np.sin(radec[1] * np.pi / 180.0)

    # -- (u,v) coordinates in m
    u = ch_ * b[0] - sl_ * sh_ * b[1] + cl_ * sh_ * b[2]
    v = (
        sd_ * sh_ * b[0]
        + (sl_ * sd_ * ch_ + cl_ * cd_) * b[1]
        - (cl_ * sd_ * ch_ - sl_ * cd_) * b[2]
    )
    return u, v
