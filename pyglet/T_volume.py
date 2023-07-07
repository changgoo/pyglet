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

import yt
import xarray as xr

if __name__ == "__main__":
    # yt.enable_parallelism()

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

    s = LoadSim(basedir)

    nums = s.nums["out2"][1:]  # skip 0
    foutdir = osp.join(os.fspath(s.basedir), "Tpdf")

    if COMM.rank == 0:
        print("basedir, nums", s.basedir, nums)
        nums = split_container(nums, COMM.size)
        os.makedirs(foutdir, exist_ok=True)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print("[rank, mynums]:", COMM.rank, mynums)

    time0 = time.time()

    # nums = s.nums["out2"][1:]  # skip 0
    # for num in nums:
    for num in mynums:
        ds = s.load_athdf(num=num, output_id=2, load_method="yt")
        ad = ds.all_data()
        ad["gas", "temperature"][ad["gas", "temperature"].to("K") > 1.0e4]

        prof = yt.create_profile(
            ad,
            [("gas", "temperature")],
            [("gas", "cell_volume")],
            weight_field=None,
            n_bins=512,
            extrema=dict(temperature=(1.0e4, 1.0e9)),
        )

        da = xr.DataArray(prof[("gas", "cell_volume")], coords=[prof.x], dims=["T"])

        fout = osp.join(foutdir, f"Tpdf_{num:04d}.nc")
        da.assign_coords(time=ds.current_time).to_netcdf(fout)

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()
