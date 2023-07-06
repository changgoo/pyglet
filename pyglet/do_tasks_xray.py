#!/usr/bin/env python

import os
import os.path as osp
import gc
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
import pprint
import argparse
import sys
import numpy as np

from pyglet.loadsim import LoadSim
from pyglet.xray import Xray

from pyglet.utils.split_container import split_container
from pyglet.utils.make_movie import make_movie

import yt, soxs, pyxsim

if __name__ == "__main__":
    COMM = MPI.COMM_WORLD

    basedir_def = "/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR"

    savdir = None
    savdir_pkl = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", type=str, default=basedir_def, help="Name of the basedir."
    )
    args = vars(parser.parse_args())
    locals().update(args)

    import astropy.constants as ac
    redshift=0.00001129
    # from astropy.cosmology import Planck18, z_at_value

    # rmax = 100 * ac.pc
    # dist = 50 * ac.kpc  ## LMC distance
    # redshift = z_at_value(
    #     Planck18.comoving_distance, dist, zmax=0.1, method="bounded"
    # ).value
    # ang = np.rad2deg((rmax / dist).cgs.value) * 60

    s = LoadSim(basedir)

    nums = s.nums["out2"][1:]  # skip 0

    if COMM.rank == 0:
        print("basedir, nums", s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print("[rank, mynums]:", COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        print(num, end=" ")

        ds = s.load_athdf(num=num, output_id=2, load_method="yt")
        xray = Xray(
            ds,
            emax=3.0,
            binscale="linear",
            nbins=10000,
            savdir=os.fspath(s.basedir),
            redshift=redshift,
        )
        xray.set_regions()
        # for ax in ['x','y','z']:
        for ax in ["z"]:
            xray.set_filenames(axis=ax, photon=True, event=True)
            xray.create_simput(axis=ax, overwrite=False)
            xray.clean_up(event=True)
            for inst in ["lem_2eV", "lem_0.9eV"]:
                for exp in [10, 100]:
                    xray.instrument_simulator(exp=exp, inst=inst, overwrite=False)
                    # xray.clean_up(inst=True)
        # varying exposure time
        f1 = xray.show(
            match=dict(dist="50kpc", src="onsrc", axis="z", inst="lem_2eV"),
            vmax=None,
            vmin=None,
        )
        f1.savefig(
            os.path.join(
                xray.figdir,
                "{}_xray_figure_exptime_{}.png".format(xray.basename, "lem_2eV"),
            ),
            bbox_inches="tight",
        )
        # varying exposure time
        f2 = xray.show(
            match=dict(dist="50kpc", src="onsrc", axis="z", inst="lem_0.9eV"),
            vmax=None,
            vmin=None,
        )
        f2.savefig(
            os.path.join(
                xray.figdir,
                "{}_xray_figure_exptime_{}.png".format(xray.basename, "lem_0.9eV"),
            ),
            bbox_inches="tight",
        )
        # varying targets
        f3 = xray.show(
            match=dict(dist="50kpc", axis="z", exptime="100ks", inst="lem_2eV"),
            vmax=None,
            vmin=None,
        )
        f3.savefig(
            os.path.join(
                xray.figdir,
                "{}_xray_figure_target_{}.png".format(xray.basename, "lem_2eV"),
            ),
            bbox_inches="tight",
        )
        # varying targets
        f4 = xray.show(
            match=dict(dist="50kpc", axis="z", exptime="100ks", inst="lem_0.9eV"),
            vmax=None,
            vmin=None,
        )
        f4.savefig(
            os.path.join(
                xray.figdir,
                "{}_xray_figure_target_{}.png".format(xray.basename, "lem_0.9eV"),
            ),
            bbox_inches="tight",
        )
        plt.close("all")
        xray.clean_up(photon=True)

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()

    # COMM.barrier()

    if COMM.rank == 0:
        fin = osp.join(os.fspath(s.basedir), "xray/figure/*_xray_figure_exptime.png")
        fout = osp.join(os.fspath(s.basedir), "{0:s}_xray_exptime.mp4".format(basename))
        make_movie(fin, fout, fps_in=15, fps_out=15)
        from shutil import copyfile

        copyfile(
            fout,
            osp.join(
                "/tigress/changgoo/public_html/temporary_movies/SNTI",
                osp.basename(fout),
            ),
        )
