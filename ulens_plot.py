import cmath
import io
from copy import deepcopy

import corner
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ulens_utils as ut
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


def multipage(filename, figs=None):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format="pdf")
    pp.close()


def rasterize(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    buf.seek(0)
    pil_img = deepcopy(Image.open(buf))
    buf.close()
    return pil_img


def draw(x, y, layers, nsigma, colors, marker="."):
    subfig = plt.figure(frameon=False, figsize=(2.5, 2.5))
    mpl.font_manager._get_font.cache_clear()  # necessary to reduce text corruption artifacts
    axes = plt.axes()
    axes.set_axis_off()
    for i in range(nsigma):
        _x, _y = layers[i][y + 1], layers[i][x + 1]
        axes.scatter(_x, _y, fc=colors[nsigma - 1 - i], s=10, ec="none", marker=marker)
    pil_img = rasterize(subfig)
    plt.close()
    return pil_img, axes.get_xlim(), axes.get_ylim()


def cmp_region(
    fs,
    fig,
    parameters,
    hist_kwargs={},
    tex=None,
    labels=None,
    colors=["red", "blue", "lime", "gold", "cyan", "magenta", "olivedrab"],
    lss=None,
    modfunc=None,
):
    tex = tex if tex else parameters
    for i, fn in enumerate(fs):
        tab = pd.read_csv(fn)[:]
        if modfunc is not None:
            tab = modfunc(tab)
        m = min(tab["chi2"])
        tab = tab[tab["chi2"] < m + 49]
        tab = tab[parameters].to_numpy()
        contour_kwargs = {"linewidths": 0.7}
        hist_kwargs.update({"color": colors[i], "density": True, "linewidth": 0.7})
        if lss:
            contour_kwargs["linestyles"] = lss[i]
            hist_kwargs["linestyle"] = lss[i]
        corner.corner(
            tab,
            plot_density=False,
            labels=tex,
            no_fill_contours=True,
            plot_datapoints=False,
            smooth=1,
            color=colors[i],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
            fig=fig,
            hist_kwargs=hist_kwargs,
            hist2d_kwargs={"contour_kwargs": {"linewidths": 0.7}},
            contour_kwargs=contour_kwargs,
        )
        # labels[i] += rf"; $\chi^2={m:.1f}$"

    fig.axes[len(parameters)-1].legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=labels[i], ls=lss[i])
            for i in range(len(fs))
        ],
        fontsize=14,
        frameon=False,
        loc="upper right",
    )
    return fig


def chi2plot(
    tab,
    fig,
    parameters,
    colorbar=True,
    bins=20,
    nsigma=4,
    delta=1,
    filename=None,
    hist_kwargs={},
    s1="percentile",
    tex=None,
    colors=["red", "gold", "limegreen", "blue", "gray", "lightblue"],
):
    plt.rcParams["font.size"] = "12"
    n_param = len(parameters)
    gs = fig.add_gridspec(n_param, n_param, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=False, sharey=False)
    sample = tab[["chi2"] + parameters]
    sample = np.lib.recfunctions.structured_to_unstructured(sample.as_array())

    chi2 = min(sample[:, 0])
    print(chi2)
    sample[:, 0] -= chi2
    sample = sample[sample[:, 0] < ((nsigma) ** 2) * delta]

    layers = []
    for i in range(nsigma):
        sample = sample[sample[:, 0] < (nsigma - i) ** 2 * delta]
        layers.append(sample.T)

    from multiprocessing import Pool

    args = []
    for x in range(n_param):
        for y in range(x):
            args.append((x, y, layers, nsigma, colors))
    for x in range(n_param):
        axs[x][x].hist(
            layers[0][x + 1], bins=bins, density=True, alpha=0.4, edgecolor="none", lw=0
        )
        if s1 == "percentile":
            lw, ct, up = np.percentile(layers[0][x + 1], [16, 50, 84])
        elif s1 == "mid":
            up = max(layers[-1][x + 1])
            lw = min(layers[-1][x + 1])
            ct = (up + lw) / 2
        else:
            ct = layers[-1][x + 1][0]
            up = max(layers[-1][x + 1])
            lw = min(layers[-1][x + 1])
        axs[x][x].axvline(ct, color="k")
        axs[x][x].axvline(lw, ls="--", color="k")
        axs[x][x].axvline(up, ls="--", color="k")
        n = parameters[x]
        if tex:
            n = tex[n]
        title = f"{n}\n${ct:.3f}_{{{lw-ct:.3f}}}^{{+{up - ct:.3f}}}$"
        axs[x][x].set_title(title, fontsize=12)

    axdict = {par: {} for par in parameters}
    with Pool(int(n_param * (n_param - 1) / 2)) as p:
        for arg, output in zip(args, p.starmap(draw, args)):
            rastered, xlim, ylim = output
            axs[arg[0]][arg[1]].imshow(rastered, extent=[*xlim, *ylim], origin="upper")
            axs[arg[0]][arg[1]].set_aspect("auto")
            axs[arg[1]][arg[1]].set_xlim(*xlim)
            axdict[parameters[arg[1]]][parameters[arg[0]]] = axs[arg[0]][arg[1]]

    for x in range(n_param):
        axs[x][0].tick_params(axis="both", which="major")
        axs[n_param - 1][x].tick_params(axis="both", which="major")
        axs[x][0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        axs[n_param - 1][x].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        axs[n_param - 1][x].set_xticks(axs[n_param - 1][x].get_xticks()[1:-1])
        axs[x][0].set_yticks(axs[x][0].get_yticks()[1:-1])
        plt.setp(axs[n_param - 1][x].xaxis.get_majorticklabels(), rotation=45)
        plt.setp(axs[x][0].yaxis.get_majorticklabels(), rotation=45)
    for ax in axs.flat:
        ax.label_outer()
    for x in range(n_param):
        for y in range(x + 1, n_param):
            axs[x][y].set_visible(False)

    if colorbar:
        fig.tight_layout()
        fig.subplots_adjust(right=0.9)
        cmap = mpl.colors.ListedColormap(colors[:nsigma])
        bounds = delta * np.arange(nsigma + 1) ** 2
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=fig.add_axes([0.92, 0.07, 0.03, 0.88]),
        )
        cbar.minorticks_off()
    plt.suptitle(r"min $\chi^2$: " + f"{chi2:.2f}")
    if filename:
        plt.savefig(filename)
    return fig, axdict


def caustics(s, q, nrot=200, resample=True):
    """
    e.g.
    critic, caustic = caustics(s, q, nrot=200, resample=True)
    for c in caustic:
        plt.plot(c.real, c.imag)
    """
    z1 = -s / 2
    z2 = s / 2
    m1 = 1 / (1 + q)
    m2 = 1 - m1
    # check type : 1 - wide , 0 resonate , -1 close
    t = 0
    if s > dw(q):
        t = 1
        _key = lambda x: -cmath.phase(x[1])
    elif s < dc(q):
        t = -1
        _key = lambda x: -x[1].imag
    phi_l = np.linspace(0, 2 * np.pi * (1 - 1 / nrot), nrot)
    caustics = []
    critical = []
    for phi in phi_l:
        ephi_c = cmath.exp(1.0j * phi)
        coeff = [
            -(m1 * z2**2.0 + m2 * z1**2.0) * ephi_c + z1**2.0 * z2**2.0,
            (2.0 * z2 * m1 + 2.0 * z1 * m2) * ephi_c,
            2.0 * z1 * z2 - ephi_c,
            0.0,
            1.0,
        ]
        solves = np.polynomial.polynomial.polyroots(coeff)
        tcaus = np.array([lenseq(i, z1, z2, m1, m2) for i in solves])
        if phi == 0:
            if t == -1:
                solves, tcaus = zip(
                    *sorted(zip(solves, tcaus), key=lambda x: -x[1].imag)
                )
                solves, tcaus = list(solves), list(tcaus)
                if tcaus[1].real > tcaus[2].real:
                    solves[1], solves[2] = solves[2], solves[1]
                    tcaus[1], tcaus[2] = tcaus[2], tcaus[1]
            if t == 0:
                r = sorted(solves, key=lambda x: x.imag)
                _key = lambda x: -cmath.phase(x[1] - center)
                center = (r[1] + r[2]) / 2
                solves, tcaus = zip(
                    *sorted(zip(solves, tcaus), key=lambda x: x[1].imag)
                )
                solves, tcaus = list(solves), list(tcaus)
                order = [1, 3, 2, 0]
                if tcaus[1].real > tcaus[2].real:
                    order[0], order[2] = 2, 1
                solves, tcaus = [solves[i] for i in order], [tcaus[i] for i in order]
            if t == 1:
                solves, tcaus = zip(
                    *sorted(zip(solves, tcaus), key=lambda x: (x[1].real))
                )
                order = [0, 2, 3, 1]
                solves, tcaus = [solves[i] for i in order], [tcaus[i] for i in order]
        else:
            solves, tcaus = zip(*sorted(zip(solves, tcaus), key=_key))
            solves, tcaus = list(solves), list(tcaus)
        critical.append(solves)
        caustics.append(tcaus)
    cut = 0
    critical = np.array(critical[cut:] + critical[:cut]).T
    caustics = np.array(caustics[cut:] + caustics[:cut]).T
    if not resample:
        return critical, caustics
    if t == 0:
        # resonate type
        # input : phi_l ,
        # image list :
        _phi = np.concatenate([phi_l, phi_l + 2 * np.pi])
        _critic = np.concatenate([critical[0], critical[1]])
        freal = interp1d(
            _phi,
            _critic.real,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        fimag = interp1d(
            _phi,
            _critic.imag,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        # plt.scatter(_phi,_critic.real)
        # plt.show()
        _dzdphi = dzdphi(_critic, s, q)
        _dzetadphi = abs(_dzdphi + np.exp(1j * _phi) * _dzdphi.conjugate())
        fdsdphi = interp1d(
            _phi, _dzetadphi, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )

        __phi = np.linspace(0, 4 * np.pi, 10 * nrot)
        __s = cumtrapz(fdsdphi(__phi), __phi, initial=0)
        fphis = interp1d(
            __s, __phi, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )

        __s_resample = np.linspace(__s[0], __s[-1], nrot)
        __phi_resample = fphis(__s_resample)

        nreal = freal(__phi_resample)
        nimag = fimag(__phi_resample)
        ncritical = np.append((nreal + nimag * 1j), np.flip(nreal + nimag * -1j))
        ncaustics = lenseq(ncritical, z1, z2, m1, m2)
        return [ncritical], [ncaustics]
    else:
        _phi = phi_l
        _critic1, _critic2 = critical[0], critical[1]
        freal1 = interp1d(
            _phi,
            _critic1.real,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        fimag1 = interp1d(
            _phi,
            _critic1.imag,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        freal2 = interp1d(
            _phi,
            _critic2.real,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        fimag2 = interp1d(
            _phi,
            _critic2.imag,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        _dzdphi1 = dzdphi(_critic1, s, q)
        _dzetadphi1 = abs(_dzdphi1 + np.exp(1j * _phi) * _dzdphi1.conjugate())
        fdsdphi1 = interp1d(
            _phi,
            _dzetadphi1,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        _dzdphi2 = dzdphi(_critic2, s, q)
        _dzetadphi2 = abs(_dzdphi2 + np.exp(1j * _phi) * _dzdphi2.conjugate())
        fdsdphi2 = interp1d(
            _phi,
            _dzetadphi2,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

        __phi = np.linspace(0, 2 * np.pi, 10 * nrot)
        __s1 = cumtrapz(fdsdphi1(__phi), __phi, initial=0)
        __s2 = cumtrapz(fdsdphi2(__phi), __phi, initial=0)
        fphis1 = interp1d(
            __s1, __phi, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )
        fphis2 = interp1d(
            __s2, __phi, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )

        __s_resample1 = np.linspace(__s1[0], __s1[-1], nrot)
        __phi_resample1 = fphis1(__s_resample1)

        __s_resample2 = np.linspace(__s2[0], __s2[-1], nrot)
        __phi_resample2 = fphis2(__s_resample2)

        nreal1 = freal1(__phi_resample1)
        nimag1 = fimag1(__phi_resample1)
        nreal2 = freal2(__phi_resample2)
        nimag2 = fimag2(__phi_resample2)
        if t == 1:
            ncritical1 = np.concatenate(
                [nreal1 + nimag1 * 1j, np.flip(nreal1 + nimag1 * -1j)]
            )
            ncritical2 = np.concatenate(
                [nreal2 + nimag2 * 1j, np.flip(nreal2 + nimag2 * -1j)]
            )
            ncaustics1 = lenseq(ncritical1, z1, z2, m1, m2)
            ncaustics2 = lenseq(ncritical2, z1, z2, m1, m2)
            return [ncritical1, ncritical2], [ncaustics1, ncaustics2]
        else:
            ncritical1p = nreal1 + nimag1 * 1j
            ncritical1n = nreal1 + nimag1 * -1j
            ncritical2 = np.concatenate(
                [nreal2 + nimag2 * 1j, np.flip(nreal2 + nimag2 * -1j)]
            )
            ncaustics1p = lenseq(ncritical1p, z1, z2, m1, m2)
            ncaustics1n = lenseq(ncritical1n, z1, z2, m1, m2)
            ncaustics2 = lenseq(ncritical2, z1, z2, m1, m2)
            return [ncritical1p, ncritical1n, ncritical2], [
                ncaustics1p,
                ncaustics1n,
                ncaustics2,
            ]


#    dzetadphi = cmath.exp(j*phi)
# the caustics sastify the eq : m1/(z-z1)**2 + m2/(z-z2)**2 = exp(-i*phi)
# however the z is in the image plane
# we try to define a paramter s such that d zeta(s)/d s is an constant
# we have a parameterization of the caustics with zeta(phi) and we try to find the relation between phi and s


def dzdphi(z, s, q):
    z1 = complex(-s / 2, 0)
    z2 = complex(s / 2, 0)
    m1 = 1 / (1 + q)
    m2 = 1 - m1
    return (
        1j
        / 2
        * (m2 * (z - z1) ** 2 + m1 * (z - z2) ** 2)
        / (m2 * (z - z1) ** 3 + m1 * (z - z2) ** 3)
        * (z - z1)
        * (z - z2)
    )


def lenseq(z, z1, z2, m1, m2):
    zeta_c = z.conjugate() + m1 / (z1 - z) + m2 / (z2 - z)
    return zeta_c.conjugate()


def dc(q):
    # dc^8 = (1+q)^2/27/q*(1-dc^4)^3
    c = (1 + q) ** 2 / 27 / q
    dc4 = np.polynomial.polynomial.polyroots([-c, 3 * c, 1 - 3 * c, c])
    for i in dc4:
        if np.isreal(i):
            return i.real**0.25
    return "no solve found , pls check input"


def dw(q):
    return ((1 + q ** (1 / 3)) ** 3 / (1 + q)) ** 0.5


def plot2dfunc(x, y, f, ax):
    suv = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            suv[j, i] = f(x[i], y[j])
    return ax.pcolormesh(x, y, suv)


def plot_traj(
    param,
    ax,
    t_model=np.linspace(9200, 9800, 2000),
    ap=[],
    npoints=2000,
    t_arrow=None,
    arr_size=1.0,
    ckwargs={},
    tkwargs={},
    sckwargs={},
    hs=False,
    sc=False,
):
    from ulens_models import Binary_orbital3d_traj

    param_noorb = param.copy()
    param_noorb["w1"] = 0
    param_noorb["w2"] = 0
    param_noorb["w3"] = 0
    _traj_ref = Binary_orbital3d_traj(t_model, param_noorb)
    _traj = Binary_orbital3d_traj(t_model, param)

    def ang(v0, v1):
        return np.arctan2(np.linalg.det([v0, v1]), np.dot(v0, v1))

    def rotate(v1, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        return np.dot(R, v1)

    angs = [ang(v1, v2) for v1, v2 in zip(_traj[:2].T, _traj_ref[:2].T)]
    xs, ys = np.array([rotate(v, _ang) for v, _ang in zip(_traj[:2].T, angs)]).T
    # vv = np.array([rotate(v, a) for v, a in zip(vv, ang)]).T

    rho = param["rhos"]
    ax.plot(xs, ys, **tkwargs, lw=0.7)
    if t_arrow is not None:
        xt, yt = Binary_orbital3d_traj([t_arrow - 0.1, t_arrow + 0.1], param)[:2]
        xm = np.mean(xt)
        ym = np.mean(yt)
        offx = xt[1] - xt[0]
        offy = yt[1] - yt[0]
        ax.arrow(
            xm,
            ym,
            offx,
            offy,
            color="black",
            lw=0.5,
            length_includes_head=True,
            head_width=0.03 * arr_size,
            overhang=0.9,
        )
    else:
        ax.arrow(
            (xs[int(npoints * 2 / 3) + 1] + xs[int(npoints * 2 / 3)]) / 2,
            (ys[int(npoints * 2 / 3) + 1] + ys[int(npoints * 2 / 3)]) / 2,
            xs[int(npoints * 2 / 3) + 1] - xs[int(npoints * 2 / 3)],
            ys[int(npoints * 2 / 3) + 1] - ys[int(npoints * 2 / 3)],
            color="black",
            lw=0.5,
            length_includes_head=True,
            head_width=0.03 * arr_size,
            overhang=0.9,
        )
    ax.axis("equal")
    _traj_ref = Binary_orbital3d_traj(ap, param_noorb)
    _traj = Binary_orbital3d_traj(ap, param)
    angles = [ang(v1, v2) for v1, v2 in zip(_traj[:2].T, _traj_ref[:2].T)]

    xs, ys = np.array([rotate(v, _ang) for v, _ang in zip(_traj[:2].T, angles)]).T
    seps = _traj[2]
    for _xs, _ys, _sep, _ap, _angle in zip(xs, ys, seps, ap, angles):
        idx = ap.index(_ap)
        # _alpha = (idx + 1) / len(ap)
        _alpha = 1
        circle = plt.Circle(
            (_xs, _ys),
            rho,
            color="r",
            fill=False,
            linewidth=0.8,
            alpha=_alpha,
            **sckwargs,
        )
        if sc:
            ax.scatter(_xs, _ys, s=10, fc="r", alpha=_alpha, ec="none")
        ax.add_patch(circle)
        _, cs = caustics(_sep, param["q"], resample=1, nrot=1000)

        offset_x = _sep * (-0.5 + 1 / (1 + param["q"]))  # if s < 1 else s/2-q/(1+q)/s
        offset_y = 0
        for i in cs:
            real = i.real + offset_x
            imag = i.imag + offset_y
            real, imag = np.array([rotate(x, _angle) for x in np.c_[real, imag]]).T
            ax.plot(real, imag, **ckwargs, alpha=_alpha, lw=0.7)
        if hs:
            ax.scatter(-_sep / 2 + offset_x, offset_y, s=50, c="b", ec="none")
            ax.scatter(_sep / 2 + offset_x, offset_y, s=5, c="b", ec="none")
    return xs, ys


def plotorbi(param, c="k"):
    fig, axs = plt.subplots(3)

    def getall(w1, w2, w3):
        w13 = w1 * w1 + w3 * w3
        w123 = np.sqrt(w13 + w2 * w2)
        w13 = np.sqrt(w13)
        w = w3 * w123 / w13
        P = abs(2 * np.pi / w)
        inc = np.arccos(w2 * w3 / w13 / w123)
        phi = np.arctan(-w1 * w123 / w3 / w13)
        return inc, phi, P

    w1, w2, w3 = param["w1"], param["w2"], param["w3"]
    inc, phi0, P = getall(w1, w2, w3)

    t = np.array([9865, 9895, 10001.8])
    phi = (t - 10001.8) / P * 2 * np.pi + phi0
    x = np.cos(phi)
    y = np.sin(phi)
    alpha = np.arctan2(y * np.cos(inc), x)
    alpha0 = np.arctan2(y[2] * np.cos(inc), x[2]) - param["alpha"]
    s0 = (x[2] ** 2 + y[2] ** 2 * np.cos(inc) ** 2) ** 0.5 / param["s"]
    x /= s0
    y /= s0

    axs[0].scatter(x, y * np.cos(inc), c=c, s=50)
    axs[1].scatter(t, (x**2 + y**2 * np.cos(inc) ** 2) ** 0.5, c=c)
    axs[2].scatter(t, (alpha - alpha0) % (2 * np.pi), c=c, s=50)

    t = np.linspace(9800, 9800 + P, 10000)
    phi = (t - 10001.8) / P * 2 * np.pi + phi0
    x = np.cos(phi)
    y = np.sin(phi)
    x /= s0
    y /= s0

    axs[0].plot(x, y * np.cos(inc), c + "--")

    t = np.linspace(9800, 10100, 10000)
    phi = (t - 10001.8) / P * 2 * np.pi + phi0
    x = np.cos(phi)
    y = np.sin(phi)
    alpha = np.arctan2(y * np.cos(inc), x)
    x /= s0
    y /= s0

    axs[0].scatter(x, y * np.cos(inc), c=t, s=1)
    axs[1].plot(t, (x**2 + y**2 * np.cos(inc) ** 2) ** 0.5, c=c)
    axs[2].plot(t, (alpha - alpha0) % (2 * np.pi), c=c)
    axs[0].axis("equal")
    return fig

def plot_lc(
    param: dict,
    phots: list,
    idref: int,
    axs: list,
    modelfunc: callable,
    args_list=None,
    plot_points=True,
    t_model=np.linspace(9000, 9800, 2000),
    zorder=None,
    ap=[],
    clist=[
        "r",
        "black",
        "darkgreen",
        "blue",
        "orange",
        "magenta",
        "lime",
        "olivedrab",
        "darkslategray",
        "r",
        "lime",
        "gold",
    ],
    labels=None,
    sigma=2.5,
    fancy_lg=False,
    plot_mask=False,
    bands=[],
    modelplot_kwargs={},
    verbose=True,
    alpha:float=1,
) -> list:
    """
    INPUTS:
        param    : dict, model parameters to be put in modelfunc
        phots    : list, list of photometry data
        idref    : int, index of the reference photometry system
        axs      : [ax] or [ax, axr], where axr is the residual plot
        modelfunc: callable, model function, modelfunc(t, param, *args)
        args_list: list, list of args to be put in modelfunc (this can be different for different photmetry, like limb-darkening coeff)
        t_model  : time_span to plot the model
        clist    : list, list of colors for each photometry
        labels   : list, list of labels for each photometry, if None it will be the name of the photometry
        sigma    : float, threshold to reject points in data
        fancy_lg : bool, if True, 2 columns of legend will be used (labels + bands)
        bands    : list, list of bands for each photometry

    OUTPUTS:
        axs      : list of ax
        chi2s    : list of chi2 for each photometry file for each point
        residuals: list of residuals for each photometry
        chi2tot  : total chi2
    """
    if zorder is None:
        zorder = [i for i in range(len(phots))]
    chi2tot: float = 0
    chi2s = []
    residuals = []
    if labels is None:
        labels = [phot["name"].split(".")[0] for phot in phots]
    if args_list is None:
        args_list = [()] * len(phots)

    _fl = phots[idref]["flux"]
    _t_obs = _fl[:, 0]
    _A = modelfunc(_t_obs, param, *args_list[idref])
    _, _f = ut.getchi2_single(_fl, _A, phots[idref]["blending"])
    fs_ref, fb_ref = _f[:2]

    _A = modelfunc(t_model, param, *args_list[idref])

    mag_model = ut.flux2mag(_A * fs_ref + fb_ref) 
    axs[0].plot(t_model, mag_model, color="black", lw=0.7, **modelplot_kwargs, zorder=10000)


    if len(ap):
        print(modelfunc(ap, param, *args_list[idref]))

    for i, phot in enumerate(phots):
        fl = phot["flux"]
        t_obs = fl[:, 0]
        A = modelfunc(t_obs, param, *args_list[i])
        c, f = ut.getchi2_single(fl, A, phot["blending"])
        name = phot["name"]
        if verbose:
            print(
                f"{name:16s}  ",
                "chi2=",
                f"{c:8.2f}  ",
                "chi2/dof=",
                f"{c/(len(t_obs)-2):4.2f}  ",
                f"fs={f[0]:8.1f}  fb={f[1]:8.1f}",
            )
        chi2list = (fl[:, 1] - f[0] * A - f[1]) ** 2 / fl[:, 2] ** 2
        chi2s.append(chi2list)

        fl_out = fl.copy()
        fl_out[:, 1] = (fl[:, 1] - f[1]) / f[0] * fs_ref + fb_ref
        fl_out[:, 2] = fl[:, 2] / f[0] * fs_ref

        mag_out = ut.flux2mag(fl_out)
        mag_model = ut.flux2mag(A * fs_ref + fb_ref)
        if plot_points:
            axs[0].errorbar(
                mag_out[:, 0],
                mag_out[:, 1],
                yerr=mag_out[:, 2],
                fmt="o",
                fillstyle="none",
                color=clist[i],
                capsize=0,
                zorder=zorder[i],
                markersize=4,
                alpha=alpha,
            )
        residuals.append(mag_out[:, 1] - mag_model)

        if len(axs) > 1:
            axs[1].errorbar(
                mag_out[:, 0],
                mag_out[:, 1] - mag_model,
                yerr=mag_out[:, 2],
                fmt="o",
                fillstyle="none",
                color=clist[i],
                capsize=0,
                zorder=zorder[i],
                markersize=4,
                alpha=alpha,
            )
        if "mask" in phot and plot_mask:
            mask = phot["mask_data"]
            mask_flux = ut.mag2flux(mask)
            A = modelfunc(mask[:, 0], param, *args_list[i])
            mask_model = ut.flux2mag(A * fs_ref + fb_ref)
            mask_flux[:, 1] = (mask_flux[:, 1] - f[1]) / f[0] * fs_ref + fb_ref
            mask_flux[:, 2] = mask_flux[:, 2] / f[0] * fs_ref
            mask = ut.flux2mag(mask_flux)
            axs[0].errorbar(
                mask[:, 0],
                mask[:, 1],
                yerr=mask[:, 2],
                fmt="x",
                fillstyle="none",
                color="gray",
                markersize=6,
                zorder=10000
            )
            if len(axs) > 1:
                axs[1].errorbar(
                    mask[:, 0],
                    mask[:, 1] - mask_model,
                    yerr=mask[:, 2],
                    fmt="x",
                    fillstyle="none",
                    color="gray",
                    markersize=6,
                    zorder=10000
                )

        if "mask" in phot:
            raw_flux = ut.mag2flux(phot["raw_data"])
            raw_model = modelfunc(raw_flux[:, 0], param, *args_list[i])
            chi2list = (raw_flux[:, 1] - f[0] * raw_model - f[1]) ** 2 / raw_flux[
                :, 2
            ] ** 2

        mask = np.zeros_like(chi2list, dtype=bool)
        if "mask" in phot:
            mask[phot["mask"]] = True
        chi2list = np.ma.array(chi2list, mask=mask)
        thres = 1e16
        flt = (
            (chi2list > np.mean(chi2list) * sigma ** 2) | (phot["raw_data"][:, 2] > thres)
        ) & (~chi2list.mask)
        idx = np.where(flt)[0]
        n = np.sum(1*mask)
        n0 = np.sum(1*mask)
        # if verbose:
        #     print(idx, n)
        #     print(max(chi2list))
        #     print(np.mean(chi2list))
        while len(idx) > n:
            chi2list.mask[idx] = True
            flt = (
                (chi2list > np.mean(chi2list) * sigma ** 2)
                | (phot["raw_data"][:, 2] > thres)
            ) & (~chi2list.mask)
            idx = np.where(flt)[0]
            n = len(idx)
        mask_incres = list(np.where(chi2list.mask == True)[0])
        if verbose:
            if len(mask_incres) > n0:
                print("\tsuggested mask & escale")
                print("\tmask=", repr(mask_incres))
            ec = (c / (np.sum(1 * (~chi2list.mask)) - 2)) ** 0.5 * phot["escale"]
            print("\tescale=", f"{ec:4.2f}")

        chi2tot += c

        if len(axs) > 1:
            axs[1].axhline(0, linewidth=0.7, **modelplot_kwargs, zorder=10000)
    nc = 0
    for i, phot in enumerate(phots):
        if not labels[i]:
            continue
        axs[0].errorbar(
            [],
            [],
            yerr=[],
            fmt="o",
            label=labels[i],
            color=clist[i],
            capsize=0,
            fillstyle="none",
        )
        nc += 1
    if fancy_lg:
        for i, phot in enumerate(phots):
            if not labels[i]:
                continue
            axs[0].errorbar(
                [],
                [],
                yerr=[],
                fmt="o",
                label=bands[i],
                color=clist[i],
                capsize=0,
                fillstyle="none",
            )
        legend = axs[0].legend(
            handlelength=0,
            handleheight=1,
            markerscale=0,
            frameon=False,
            ncol=2,
            columnspacing=0.05,
            labelspacing=0.17,
            fontsize=16,
        )
        cs = clist[:nc] + clist[:nc]
        for n, text in enumerate(legend.texts):
            text.set_color(cs[n])
        for item in legend.legend_handles:
            item.set_visible(False)
    else:
        legend = axs[0].legend(fontsize=15)

    print(f"chi2tot={chi2tot:12.2f}")
    return axs, chi2s, chi2tot, residuals
