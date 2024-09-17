import numpy as np


def lenseq(z, z1, z2, m1, m2):
    zeta_c = z.conjugate() + m1 / (z1 - z) + m2 / (z2 - z)
    return zeta_c.conjugate()


def binary_images(s, q, ux, uy, rhos, grid_size=0.0015):
    m1 = 1
    m2 = q
    z1 = -q / (1 + q) * s + 0j
    z2 = 1 / (1 + q) * s + 0j
    src = ux + uy * 1j
    _M = 2
    minX, maxX = -_M, _M
    x = np.linspace(minX, maxX, int((maxX - minX) / grid_size) + 1)
    X, Y = np.meshgrid(x, x)
    X = X.ravel()
    Y = Y.ravel()
    coords = X + Y * 1j

    zs = lenseq(coords, z1, z2, m1, m2)
    flt = (zs - src) * (zs - src).conjugate() < rhos**2
    return coords[flt].real, coords[flt].imag


def ESPL_img(u_x, u_y, rhos, grid_size=0.0015):
    u = (u_x**2 + u_y**2) ** 0.5
    _M = u + rhos + 1
    minX, maxX = -_M, _M
    x = np.linspace(minX, maxX, int((maxX - minX) / grid_size) + 1)
    X, Y = np.meshgrid(x, x)

    X = X.ravel()
    Y = Y.ravel()
    coords = np.c_[X, Y]
    R2 = (X**2 + Y**2)[:, None]
    R = R2**0.5
    flt = (np.abs(R2 - 1) <= (u + rhos) * R).T[0]
    R = R[flt]
    R2 = R2[flt]
    coords = coords[flt]
    # return coords
    U = (coords * (1 - 1 / R2)).T
    dist2 = (U[0] - u_x) ** 2 + (U[1] - u_y) ** 2
    coords = coords[dist2 <= rhos**2]
    return coords


def chang_refsdal2(shear, u_x, u_y, rhos, grid_size=0.0015):
    """
    using chang_refsdal lens with pure shear
    """
    u = (u_x**2 + u_y**2) ** 0.5
    # r ** 2 > 1/ ( 2 * (u+rhos) ** 2 )
    rr = (u + rhos) ** 2.0
    r = rr**0.5
    linvs = (1 - shear) + rr / 2.0 - r * (rr / 4.0 + (1 - shear)) ** 0.5
    uinvs = (1 + shear) + rr / 2.0 + r * (rr / 4.0 + (1 + shear)) ** 0.5
    _M, _m = 1 / linvs**0.5, 1 / uinvs**0.5
    # _m = 1 / (2 * _M - 1)
    minX, maxX = -1 - u - rhos, 1 + u + rhos
    x = np.linspace(minX, maxX, int((maxX - minX) / grid_size) + 1)
    X, Y = np.meshgrid(x, x)
    X = X.ravel()
    Y = Y.ravel()
    R2 = X**2 + Y**2
    flt = np.logical_and(R2 <= _M**2, R2 >= _m**2)
    X = X[flt]
    Y = Y[flt]
    R2 = R2[flt]
    coords = np.c_[X, Y]
    S1 = (1 - shear) * X - X / R2
    S2 = (1 + shear) * Y - Y / R2
    flt = (S1 - u_x) ** 2 + (S2 - u_y) ** 2 <= rhos**2
    return coords[flt], 1 / linvs**0.5, 1 / uinvs**0.5


def chang_refsdal_caustics(shear):
    # lens equation
    theta = np.linspace(0, 2 * np.pi, 1000)[:-1]
    a = 1 - shear**2
    b = -2 * np.cos(2 * theta) * shear
    c = -1
    r2 = 1 / (2 * a) * (-b + (b**2 - 4 * a * c) ** 0.5)
    r = r2**0.5
    x, y = r * np.cos(theta), r * np.sin(theta)
    return (1 + shear) * x - x / r2, (1 - shear) * y - y / r2


def S(k, vertices):
    N = len(vertices)
    i = np.array(range(N))
    ip1 = (i + 1) % N
    in1 = (i - 1) % N
    edge_vector = vertices[ip1] - vertices[i]
    sum_vector = (vertices[ip1] + vertices[i]) / 2
    fac = np.dot(k, k) * 1j
    sum = 0
    for i in range(N):
        t1 = np.dot([0, 0, 1], np.cross(edge_vector[i], k))
        t2 = np.exp(1j * np.dot(k, sum_vector[i])) * np.sinc(
            np.dot(k, sum_vector[i]) / 2
        )
        sum += t1*t2
    sum *= fac
    return abs(sum)
