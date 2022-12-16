import numpy as np
import xarray as xr

def to_spherical(vec, origin):
    """Transform vector components from Cartesian to spherical coordinates

    Parameters
    ----------
    vec : tuple
        Cartesian vector components (vx, vy, vz)
    origin : tuple
        Origin of the spherical coordinates (x0, y0, z0)

    Returns
    -------
    vec_sph : tuple
        Spherical vector components (v_r, v_th, v_ph)
    """
    vx, vy, vz = vec
    x0, y0, z0 = origin
    x, y, z = vx.x, vx.y, vx.z
    # calculate spherical coordinates
    R = np.sqrt((x-x0)**2 + (y-y0)**2)
    r = np.sqrt(R**2 + (z-z0)**2)
    th = np.arctan2(R, z-z0)
    ph = np.arctan2(y-y0, x-x0)
    ph = ph.where(ph>=0, other=ph + 2*np.pi)
    sin_th, cos_th = R/r, (z-z0)/r
    sin_ph, cos_ph = (y-y0)/R, (x-x0)/R
    sin_th.loc[dict(x=x0,y=y0,z=z0)] = 0
    cos_th.loc[dict(x=x0,y=y0,z=z0)] = 0
    sin_ph.loc[dict(x=x0,y=y0)] = 0
    cos_ph.loc[dict(x=x0,y=y0)] = 0
    # transform vector components
    v_r = (vx*sin_th*cos_ph + vy*sin_th*sin_ph + vz*cos_th).rename('v_r')
    v_th = (vx*cos_th*cos_ph + vy*cos_th*sin_ph - vz*sin_th).rename('v_theta')
    v_ph = (-vx*sin_ph + vy*cos_ph).rename('v_phi')
    # assign spherical coordinates
    v_r.coords['r'] = r
    v_th.coords['r'] = r
    v_ph.coords['r'] = r
    v_r.coords['th'] = th
    v_th.coords['th'] = th
    v_ph.coords['th'] = th
    v_r.coords['ph'] = ph
    v_th.coords['ph'] = ph
    v_ph.coords['ph'] = ph
    vec_sph = (v_r, v_th, v_ph)
    return vec_sph

def groupby_bins(dat, coord, edges, cumulative=False):
    """Alternative to xr.groupby_bins, which is very slow

    Arguments
    ---------
    dat : xarray.DataArray
        input dataArray
    coord : str
        coordinate name along which data is binned
    edges : array-like
        bin edges
    cumulative : bool
        if True, perform cumulative binning, e.g.,
          v_r_binned[i] = v_r( edge[0] <= r < edge[i+1] ).mean()
        to calculate average velocity dispersion within radius r

    Return
    ------
    res: xarray.DataArray
        binned array
    """
    dat = dat.transpose('z','y','x')
    fc = dat[coord].data.flatten() # flattened coordinates
    fd = dat.data.flatten() # flattened data
    bin_sum = np.histogram(fc, edges, weights=fd)[0]
    bin_cnt = np.histogram(fc, edges)[0]
    if cumulative:
        bin_sum = np.cumsum(bin_sum)
        bin_cnt = np.cumsum(bin_cnt)
    res = bin_sum / bin_cnt
    # set new coordinates at the bin center
    centers = 0.5*(edges[1:] + edges[:-1])
    res = xr.DataArray(data=res, coords={coord:centers}, name=dat.name)
    return res
