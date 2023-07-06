import yt
import pyxsim
import soxs
import os, glob
import xarray as xr
import matplotlib.pyplot as plt


class Xray(object):
    def __init__(
        self,
        ytds,
        savdir="./",
        verbose=True,
        emin=0.1,
        emax=10,
        nbins=1000,
        redshift=0.00005,
        sky_center=(45.0, 30.0),
        model="apec",
        binscale="log",
    ):
        # self.sim = sim
        self.ytds = ytds
        self.basename = "{}".format(ytds)
        self.savdir = os.path.join(savdir, "xray2")
        self.figdir = os.path.join(savdir, "xray2", "figure")
        self.profdir = os.path.join(savdir, "xray2", "profile")
        self.verbose = verbose
        self.redshift = redshift
        self.sky_center = sky_center
        os.makedirs(self.savdir, exist_ok=True)
        os.makedirs(self.figdir, exist_ok=True)
        os.makedirs(self.profdir, exist_ok=True)

        # Zmet = self.sim.par['problem']['Z_gas']
        Zmet = 1

        self.source_model = pyxsim.CIESourceModel(
            model, emin, emax, nbins, Zmet, binscale=binscale, abund_table="aspl"
        )

    def add_xray_fields(self):
        xray_fields = self.source_model.make_source_fields(self.ytds, 0.5, 2.0)
        xray_fields += self.source_model.make_source_fields(self.ytds, 0.5, 7.0)
        self.xray_fields = xray_fields

    def create_profile(self, xy="z-vz"):
        if not hasattr(self, "xray_fields"):
            self.add_xray_fields()
        fullbox = self.ytds.all_data()
        Nx, Ny, Nz = self.ytds.domain_dimensions
        le = self.ytds.domain_left_edge.v
        re = self.ytds.domain_right_edge.v
        if xy == "z-vz":
            profile = yt.create_profile(
                data_source=fullbox,
                bin_fields=[("gas", "z"), ("gas", "velocity_z")],
                fields=[
                    ("gas", "volume"),
                    ("gas", "mass"),
                    ("gas", "xray_emissivity_0.5_2.0_keV"),
                    ("gas", "xray_emissivity_0.5_7.0_keV"),
                ],
                n_bins=(Nz, 384),
                units=dict(z="pc", velocity_z="km/s", volume="pc**3", mass="Msun"),
                logs=dict(radius=False, velocity_z=False),
                weight_field=None,
                extrema=dict(z=(le[2], re[2]), velocity_z=(-1536, 1536)),
            )
        elif xy == "T-vz":
            profile = yt.create_profile(
                data_source=fullbox,
                bin_fields=[("gas", "temperature"), ("gas", "velocity_z")],
                fields=[
                    ("gas", "volume"),
                    ("gas", "mass"),
                    ("gas", "xray_luminosity_0.5_2.0_keV"),
                    ("gas", "xray_luminosity_0.5_7.0_keV"),
                ],
                n_bins=(256, 384),
                units=dict(
                    temperature="K", velocity_z="km/s", volume="pc**3", mass="Msun"
                ),
                logs=dict(radius=False, velocity_z=False),
                weight_field=None,
                extrema=dict(temperature=(10, 1.0e10), velocity_z=(-1536, 1536)),
            )

        return profile

    def convert_profile_to_dataset(self, profile):
        # convert profiles to xarray dataset
        dset = xr.Dataset()
        for (g, k), v in profile.items():
            x = 0.5 * (profile.x_bins[:-1] + profile.x_bins[1:])
            g, xf = profile.x_field
            coords = [x]
            dims = [xf]
            if hasattr(profile, "y_bins"):
                y = 0.5 * (profile.y_bins[:-1] + profile.y_bins[1:])
                g, yf = profile.y_field
                coords.append(y)
                dims.append(yf)
            da = xr.DataArray(v, coords=coords, dims=dims)
            dset[k] = da
        return dset

    def get_profile(self, overwrite=False):
        fname = os.path.join(
            self.savdir, self.ytds.basename.replace(".tar", ".profile.nc")
        )
        if os.path.isfile(fname):
            if overwrite:
                os.remove(fname)
            else:
                dset = xr.open_dataset(fname)
                return dset

        profile = self.create_profile()
        dset = self.convert_profile_to_dataset(profile)
        dset.to_netcdf(fname)

        return dset

    def set_regions(self, zcut=None):
        ds = self.ytds
        le = ds.domain_left_edge.v
        re = ds.domain_right_edge.v
        fullbox = ds.box(le, re)
        self.regions = dict(full=fullbox)
        if zcut is not None:
            topbox = ds.box([le[0], le[1], zcut], re)
            botbox = ds.box(le, [re[0], re[1], -zcut])
            self.regions.update(dict(top=topbox, bot=botbox))

    def set_filenames(self, axis="z", photon=True, event=True, simput=True):
        redshift = self.redshift
        from astropy.cosmology import Planck18

        dist_kpc = int(Planck18.comoving_distance(redshift).to("kpc").value)

        ds = self.ytds
        if not hasattr(self, "regions"):
            self.set_regions()
        photon_fnames = dict()
        for name, box in self.regions.items():
            photon_fname = os.path.join(
                self.savdir, "{}_{}_{}kpc_photons.h5".format(ds, name, dist_kpc)
            )
            photon_fnames[name] = photon_fname

        event_fnames = dict()
        simput_fnames = dict()

        for boxname, photon_fname in photon_fnames.items():
            event_fname = photon_fname.replace("photons", "{}_events".format(axis))
            event_fnames[boxname] = event_fname
            simput_fnames[boxname] = event_fname.replace("_events.h5", "_simput.fits")
        if photon:
            self.photon_fnames = photon_fnames
        if event:
            self.event_fnames = event_fnames
        if simput:
            self.simput_fnames = simput_fnames

    def make_photons(
        self, exp_time=(1000, "ks"), area=(1, "m**2"), redshift=0.00005, overwrite=False
    ):
        if (not overwrite) and hasattr(self, "photon_fnames"):
            if self.test_files(self.photon_fnames):
                return

        from astropy.cosmology import Planck18

        dist_kpc = int(Planck18.comoving_distance(redshift).to("kpc").value)

        ds = self.ytds
        if not hasattr(self, "regions"):
            self.set_regions()
        photon_fnames = dict()
        for name, box in self.regions.items():
            photon_fname = os.path.join(
                self.savdir, "{}_{}_{}kpc_photons.h5".format(ds, name, dist_kpc)
            )
            if not overwrite and os.path.isfile(photon_fname):
                if self.verbose:
                    print("photon file {} exist".format(photon_fname))
            else:
                _photons, n_cells = pyxsim.make_photons(
                    photon_fname, box, redshift, area, exp_time, self.source_model
                )
            photon_fnames[name] = photon_fname
        self.photon_fnames = photon_fnames

    def project_photons(
        self, axis="z", nH=0.02, sky_center=(45.0, 30.0), overwrite=False
    ):
        if (not overwrite) and hasattr(self, "event_fnames"):
            if self.test_files(self.event_fnames):
                return

        ds = self.ytds
        self.make_photons(redshift=self.redshift)
        event_fnames = dict()
        for boxname, photon_fname in self.photon_fnames.items():
            event_fname = photon_fname.replace("photons", "{}_events".format(axis))
            if not overwrite and os.path.isfile(event_fname):
                if self.verbose:
                    print("event file {} exist".format(event_fname))
            else:
                n_events = pyxsim.project_photons(
                    photon_fname, event_fname, axis, sky_center, nH=nH
                )
            event_fnames[boxname] = event_fname
        self.event_fnames = event_fnames

    def test_files(self, fnames):
        all_exist = True
        for name, fname in fnames.items():
            all_exist = all_exist and os.path.isfile(fname)
        if all_exist:
            if self.verbose:
                print("all files exist")
            return True
        else:
            return False

    def create_simput(self, axis="z", overwrite=False):
        ds = self.ytds
        if (not overwrite) and hasattr(self, "simput_fnames"):
            if self.test_files(self.simput_fnames):
                return

        # if not hasattr(self,'event_fnames'):
        #     self.project_photons(axis=axis, sky_center=self.sky_center)
        # else:
        #     if not self.test_files(self.event_fnames):
        self.project_photons(axis=axis, sky_center=self.sky_center)
        simput_fnames = dict()
        for boxname, event_fname in self.event_fnames.items():
            events = pyxsim.EventList(event_fname)
            events.write_to_simput(
                event_fname.replace("_events.h5", ""), overwrite=True
            )
            simput_fnames[boxname] = event_fname.replace("_events.h5", "_simput.fits")
        self.simput_fnames = simput_fnames

    def instrument_simulator(
        self,
        axis="z",
        targets=["src", "onsrc", "offsrc"],
        exp=100,
        inst="lem_2eV",
        overwrite=False,
    ):
        if not hasattr(self, "simput_fnames"):
            self.create_simput(axis=axis)
        inst_fnames = dict()
        img_fnames = dict()
        spec_fnames = dict()
        for boxname, simput_fname in self.simput_fnames.items():
            for target in ["src", "onsrc", "offsrc"]:
                if target == "src":
                    sky_center = self.sky_center
                    kwargs = dict(
                        instr_bkgnd=False, foreground=False, ptsrc_bkgnd=False
                    )
                elif target == "onsrc":
                    sky_center = self.sky_center
                    kwargs = dict(instr_bkgnd=True, foreground=True, ptsrc_bkgnd=True)
                elif target == "offsrc":
                    sky_center = (0, 0)
                    kwargs = dict(instr_bkgnd=True, foreground=True, ptsrc_bkgnd=True)
                evt_fname = simput_fname.replace(
                    "simput", "{}_{}ks_{}".format(target, exp, inst)
                )
                if not overwrite and os.path.isfile(evt_fname):
                    if self.verbose:
                        print("{} event file {} exist".format(target, evt_fname))
                else:
                    soxs.instrument_simulator(
                        simput_fname,
                        evt_fname,
                        (exp, "ks"),
                        inst,
                        sky_center,
                        overwrite=True,
                        **kwargs,
                    )
                if os.path.isfile(evt_fname):
                    img_fname = evt_fname.replace(".fits", "_img.fits")
                    spec_fname = evt_fname.replace(".fits", "_spec.pha")
                    soxs.write_image(
                        evt_fname, img_fname, emin=0.1, emax=2.0, overwrite=True
                    )
                    soxs.write_spectrum(evt_fname, spec_fname, overwrite=True)
                    key = "{}_{}".format(boxname, target)
                    inst_fnames[key] = evt_fname
                    img_fnames[key] = img_fname
                    spec_fnames[key] = spec_fname
        self.inst_fnames = inst_fnames
        self.img_fnames = img_fnames
        self.spec_fnames = spec_fnames

    def find_img_files(self):
        return glob.glob(os.path.join(self.savdir, "{}*_img.fits".format(self.ytds)))

    def find_spec_files(self):
        return glob.glob(os.path.join(self.savdir, "{}*_spec.pha".format(self.ytds)))

    def parse_filename(self, f):
        pid = "{}".format(self.ytds)
        sp = os.path.basename(f).replace(pid, "").split("_")
        _, region, dist, axis, src, exptime = sp[:6]
        inst = "_".join(sp[6:-1])
        return dict(
            region=region, dist=dist, axis=axis, src=src, exptime=exptime, inst=inst
        )

    def plot_image(
        self,
        img_file,
        hdu="IMAGE",
        stretch="linear",
        vmin=None,
        vmax=None,
        facecolor="black",
        center=None,
        width=None,
        fig=None,
        cmap=None,
        cbar=True,
        grid_spec=None,
    ):
        """
        Plot a FITS image created by SOXS using Matplotlib.

        Parameters
        ----------
        img_file : str
            The on-disk FITS image to plot.
        hdu : str or int, optional
            The image extension to plot. Default is "IMAGE"
        stretch : str, optional
            The stretch to apply to the colorbar scale. Options are "linear",
            "log", and "sqrt". Default: "linear"
        vmin : float, optional
            The minimum value of the colorbar. If not set, it will be the minimum
            value in the image.
        vmax : float, optional
            The maximum value of the colorbar. If not set, it will be the maximum
            value in the image.
        facecolor : str, optional
            The color of zero-valued pixels. Default: "black"
        center : array-like
            A 2-element object giving an (RA, Dec) coordinate for the center
            in degrees. If not set, the reference pixel of the image (usually
            the center) is used.
        width : float, optional
            The width of the image in degrees. If not set, the width of the
            entire image will be used.
        figsize : tuple, optional
            A 2-tuple giving the size of the image in inches, e.g. (12, 15).
            Default: (10,10)
        cmap : str, optional
            The colormap to be used. If not set, the default Matplotlib
            colormap will be used.

        Returns
        -------
        A tuple of the :class:`~matplotlib.figure.Figure` and the
        :class:`~matplotlib.axes.Axes` objects.
        """
        from astropy.io import fits
        from astropy.visualization.wcsaxes import WCSAxes
        from astropy import wcs
        from astropy.wcs.utils import proj_plane_pixel_scales
        from matplotlib.colors import LogNorm, Normalize, PowerNorm

        if stretch == "linear":
            norm = Normalize(vmin=vmin, vmax=vmax)
        elif stretch == "log":
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif stretch == "sqrt":
            norm = PowerNorm(0.5, vmin=vmin, vmax=vmax)
        else:
            raise RuntimeError(f"'{stretch}' is not a valid stretch!")
        with fits.open(img_file) as f:
            hdu = f[hdu]
            w = wcs.WCS(hdu.header)
            pix_scale = proj_plane_pixel_scales(w)
            if center is None:
                center = w.wcs.crpix
            else:
                center = w.wcs_world2pix(center[0], center[1], 0)
            if width is None:
                dx_pix = 0.5 * hdu.shape[0]
                dy_pix = 0.5 * hdu.shape[1]
            else:
                dx_pix = width / pix_scale[0]
                dy_pix = width / pix_scale[1]
            if fig is None:
                fig = plt.figure(figsize=(10, 10))
            if grid_spec is None:
                grid_spec = [0.15, 0.1, 0.8, 0.8]
            # fig = plt.figure(figsize=figsize)
            # ax = WCSAxes(fig, [0.15, 0.1, 0.8, 0.8], wcs=w)
            ax = fig.add_subplot(grid_spec, projection=w)
            im = ax.imshow(hdu.data, norm=norm, cmap=cmap)
            ax.set_xlim(center[0] - 0.5 * dx_pix, center[0] + 0.5 * dx_pix)
            ax.set_ylim(center[1] - 0.5 * dy_pix, center[1] + 0.5 * dy_pix)
            ax.set_facecolor(facecolor)
            if cbar:
                plt.colorbar(im)
        return fig, ax

    def select_files(self, files, match=dict(exptime="100ks")):
        flist = []
        for f in files:
            par = self.parse_filename(f)
            skip = False
            for k, v in match.items():
                if v != par[k]:
                    skip = True
            if skip:
                continue
            flist.append(f)

        return sorted(flist)

    def show(
        self,
        img_flist=None,
        spec_flist=None,
        match=dict(exptime="100ks"),
        width=0.2,
        vmin=0,
        vmax=100,
    ):
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(12, 10))

        gs = GridSpec(3, 3, figure=fig, height_ratios=[0.9, 1, 1])

        # images
        if img_flist is None:
            img_flist = self.select_files(self.find_img_files(), match=match)

        i = 0
        for f in img_flist:
            par = self.parse_filename(f)
            self.plot_image(
                f,
                stretch="sqrt",
                cmap="arbre",
                width=width,
                vmin=vmin,
                vmax=vmax,
                fig=fig,
                grid_spec=gs[0, i],
                cbar=True,
            )
            label = "{p[exptime]}-{p[src]}-{p[inst]}".format(p=par)
            plt.title(label)
            i += 1

        # spectra
        ax = fig.add_subplot(gs[1, :])
        if spec_flist is None:
            spec_flist = self.select_files(self.find_spec_files(), match=match)
        for f in spec_flist:
            par = self.parse_filename(f)
            label = "{p[exptime]}-{p[src]}-{p[inst]}".format(p=par)
            fig, ax = soxs.plot_spectrum(
                f, xmin=0.1, xmax=2.0, fig=fig, ax=ax, fontsize=None, label=label
            )
        plt.legend()

        for j, xlim in enumerate([(0.55, 0.6), (0.8, 0.85), (0.975, 1.025)]):
            ax = fig.add_subplot(gs[2, j])
            for f in spec_flist:
                fig, ax = soxs.plot_spectrum(
                    f, xmin=0.1, xmax=2.0, fontsize=None, fig=fig, ax=ax
                )
            plt.xlim(xlim)

        return fig

    def show_profile(self, zcut=0):
        import numpy as np

        prof = self.get_profile()

        prof["xray_emissivity_0.5_2.0_keV"].sel(z=slice(-np.inf, -zcut)).sum(
            dim="z"
        ).plot(label="bot")
        prof["xray_emissivity_0.5_2.0_keV"].sum(dim="z").plot(label="full")
        prof["xray_emissivity_0.5_2.0_keV"].sel(z=slice(zcut, np.inf)).sum(
            dim="z"
        ).plot(label="top")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-30, 1.0e-20)

    def do_all(self):
        with plt.style.context(
            {"figure.dpi": 150, "font.size": 10, "figure.figsize": (4, 3)}
        ):
            fig = plt.figure()
            self.show_profile()
            fig.savefig(
                os.path.join(self.profdir, "{}_xray_profile.png".format(self.basename)),
                bbox_inches="tight",
            )
            plt.close(fig)
        self.project_photons(axis="z")
        self.instrument_simulator()
        with plt.style.context({"figure.dpi": 150, "font.size": 10}):
            fig = self.show()
            fig.savefig(
                os.path.join(self.figdir, "{}_xray_figure.png".format(self.basename)),
                bbox_inches="tight",
            )
            plt.close(fig)

    def clean_up(
        self,
        photon=False,
        event=False,
        simput=False,
        inst=False,
        image=False,
        spec=False,
    ):
        if photon:
            self._remove("photon")
        if event:
            self._remove("event")
        if simput:
            self._remove("simput")
        if inst:
            self._remove("inst")
        if image:
            self._remove("img")
        if spec:
            self._remove("spec")

    def _remove(self, ftype):
        if hasattr(self, "{}_fnames".format(ftype)):
            for name, fname in getattr(self, "{}_fnames".format(ftype)).items():
                if os.path.isfile(fname):
                    if self.verbose:
                        print("removing {}".format(fname))
                    os.remove(fname)
                if ftype == "simput":
                    fname = fname.replace("simput.fits", "phlist.fits")
                    if os.path.isfile(fname):
                        if self.verbose:
                            print("removing {}".format(fname))
                        os.remove(fname)
