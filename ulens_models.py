import numpy as np
import VBBinaryLensing

VBBL = VBBinaryLensing.VBBinaryLensing()


# VBBL.t0_par = 9895
# VBBL.t0_kep = 10001.8
def ESPL(t, param, a1=0, sat=0):
    VBBL.satellite = 0
    if sat:
        VBBL.satellite = sat
    VBBL.a1 = a1
    t0, u0, tE, rho, pi1, pi2 = [
        param[i] for i in ["t0", "u0", "tE", "rho", "pi1", "pi2"]
    ]
    y1 = len(t) * [0]
    y2 = len(t) * [0]
    mag = np.array(
        VBBL.ESPLLightCurveParallax(
            [(u0), np.log(tE), t0, np.log(rho), pi1, pi2], t, y1, y2
        )
    )
    return mag


def Binary_model(t, param, plx=False, a1=0, sat=0):
    """
    param: dict with t0, u0, tE, s, q, alpha, rhos, pi1, pi2
    """
    VBBL.a1 = a1
    VBBL.satellite = sat
    t0, u0, tE = [param[i] for i in ["t0", "u0", "tE"]]
    s, q, alpha, rho = [param[i] for i in ["s", "q", "alpha", "rhos"]]
    pi1, pi2 = 0, 0
    if plx:
        pi1, pi2 = [param[i] for i in ["pi1", "pi2"]]
    y1 = np.zeros(len(t))
    y2 = np.zeros(len(t))
    vbbl_par = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0, pi1, pi2]
    return np.array(VBBL.BinaryLightCurveParallax(vbbl_par, t, y1, y2))


def Binary_model2d(t, param, plx=False, a1=0):
    """
    param: dict with t0, u0, tE, s, q, alpha, rhos, pi1, pi2
    """
    VBBL.a1 = a1
    t0, u0, tE = [param[i] for i in ["t0", "u0", "tE"]]
    s, q, alpha, rho = [param[i] for i in ["s", "q", "alpha", "rhos"]]
    dsdt, dalphadt = [param[i] for i in ["dsdt", "dalphadt"]]
    pi1, pi2 = 0, 0
    if plx:
        pi1, pi2 = [param[i] for i in ["pi1", "pi2"]]
    y1 = np.zeros(len(t))
    y2 = np.zeros(len(t))
    seps = np.zeros(len(t))
    vbbl_par = [
        np.log(s),
        np.log(q),
        u0,
        alpha,
        np.log(rho),
        np.log(tE),
        t0,
        pi1,
        pi2,
        dsdt,
        dalphadt,
    ]
    return np.array(VBBL.BinaryLightCurveOrbital2d(vbbl_par, t, y1, y2, seps))


def Binary_orbital3d(_t, theta, a1=0):
    """
    param: dict with t0, u0, tE, s, q, alpha, rhos, pi1, pi2
    """
    VBBL.a1 = a1
    _t0, _u0, _tE = [theta[i] for i in ["t0", "u0", "tE"]]
    _s, _q, _alpha, _rho = [theta[i] for i in ["s", "q", "alpha", "rhos"]]
    _w1, _w2, _w3 = [theta[i] for i in ["w1", "w2", "w3"]]
    _pi1, _pi2 = [theta[i] for i in ["pi1", "pi2"]]
    y1 = np.zeros(len(_t))
    y2 = np.zeros(len(_t))
    seps = np.zeros(len(_t))
    params = [
        np.log(_s),
        np.log(_q),
        _u0,
        _alpha,
        np.log(_rho),
        np.log(_tE),
        _t0,
        _pi1,
        _pi2,
        _w1,
        _w2,
        _w3,
    ]
    return np.array(VBBL.BinaryLightCurveOrbital(params, _t, y1, y2, seps))


def Binary_orbital3d_traj(t, param):
    t0, u0, tE = [param[i] for i in ["t0", "u0", "tE"]]
    s, q, alpha, rho = [param[i] for i in ["s", "q", "alpha", "rhos"]]
    if not "w1" in param.keys():
        w1, w2, w3 = 0, 0, 0
    else:
        w1, w2, w3 = [param[i] for i in ["w1", "w2", "w3"]]
    pi1, pi2 = [param[i] for i in ["pi1", "pi2"]]
    y1 = np.zeros(len(t))
    y2 = np.zeros(len(t))
    seps = np.zeros(len(t))
    vbbl_par = [
        np.log(s),
        np.log(q),
        u0,
        alpha,
        np.log(rho),
        np.log(tE),
        t0,
        pi1,
        pi2,
        w1,
        w2,
        w3,
    ]
    return np.array(VBBL.BinaryLightCurveOrbital_traj(vbbl_par, t, y1, y2, seps))


def Binary_model_traj(t, param, plx=False) -> np.ndarray:
    t0, u0, tE = [param[i] for i in ["t0", "u0", "tE"]]
    s, q, alpha, rho = [param[i] for i in ["s", "q", "alpha", "rhos"]]
    pi1, pi2 = 0, 0
    if plx:
        pi1, pi2 = [param[i] for i in ["pi1", "pi2"]]
    y1 = np.zeros(len(t))
    y2 = np.zeros(len(t))
    vbbl_par = [
        np.log(s),
        np.log(q),
        u0,
        alpha,
        np.log(rho),
        np.log(tE),
        t0,
        pi1,
        pi2,
    ]
    return np.array(VBBL.BinaryLightCurveParallax_traj(vbbl_par, t, y1, y2))


def PSPL(t, param, plx=False) -> np.ndarray:
    t0, u0, tE = [param[i] for i in ["t0", "u0", "tE"]]
    pi1, pi2 = 0, 0
    if plx:
        pi1, pi2 = [param[i] for i in ["pi1", "pi2"]]
    vbbl_par = [u0, np.log(tE), t0, pi1, pi2]
    y1 = np.zeros(len(t))
    y2 = np.zeros(len(t))
    A = np.array(VBBL.PSPLLightCurveParallax(vbbl_par, t, y1, y2))

    return A
