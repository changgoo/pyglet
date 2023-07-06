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

    redshift = 0.00001129
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
        ds = s.load_athdf(num=10, output_id=2, load_method="yt")
        xray = Xray(ds, savdir=os.fspath(s.basedir))
        os.makedirs(os.path.join(xray.savdir, "T-vz-profile"), exist_ok=True)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print("[rank, mynums]:", COMM.rank, mynums)

    time0 = time.time()

    overwrite = True

    for num in mynums:
        print(num, end=" ")

        ds = s.load_athdf(num=num, output_id=2, load_method="yt")
        xray = Xray(ds, savdir=os.fspath(s.basedir))
        fname = os.path.join(
            xray.savdir, "T-vz-profile", ds.basename.replace(".athdf", ".profile.nc")
        )
        print(fname)
        if os.path.isfile(fname):
            if overwrite:
                os.remove(fname)
                create = True
            else:
                dset = xr.open_dataset(fname)
                create = False
        else:
            create = True

        if create:
            xray.add_xray_fields()
            profile = xray.create_profile(xy="T-vz")
            dset = xray.convert_profile_to_dataset(profile)
            dset.to_netcdf(fname)
            print(f"{fname} created")

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()
