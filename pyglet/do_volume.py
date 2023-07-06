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


def make_sphere_volume(ds, iclick, Nrot=256, radius=(64, "pc")):
    import yt
    from yt.visualization.volume_rendering.api import Scene, create_volume_source

    sp = ds.sphere(center=ds.domain_center, radius=radius)
    sc = yt.create_scene(sp)

    source = sc[0]
    source.set_field("density")
    source.set_log(True)

    bounds = (1, 1.0e2)
    tf = yt.ColorTransferFunction((np.log10(bounds[0]), np.log10(bounds[1])))
    tf.add_layers(5, w=0.01, alpha=np.logspace(0, 3, 5), colormap="winter")

    source.tfh.tf = tf
    source.tfh.bounds = bounds
    source.tfh.plot("transfer_function_density.png")

    # source.set_transfer_function(tf)
    # source.tfh.plot("transfer_function.png", profile_field="density")

    # Add temperature

    field = "temperature"

    vol2 = create_volume_source(sp, field=field)
    vol2.use_ghost_zones = True

    bounds = (1.0e5, 1.0e6)
    tf = yt.ColorTransferFunction((np.log10(bounds[0]), np.log10(bounds[1])))
    tf.clear()
    tf.add_layers(3, 0.02, alpha=[100] * 3, colormap="autumn")

    vol2.set_transfer_function(tf)
    vol2.tfh.plot("transfer_function_temperature.png")
    sc.add_source(vol2)

    cam = sc.camera
    cam.zoom(2.0)
    cam.resolution = (1024, 1024)

    # rotate iclicks
    frame = 0
    for _ in cam.iter_rotate(2 * np.pi, Nrot):
        frame += 1
        if frame == (iclick % Nrot):
            break

    sc.annotate_domain(ds, color=[1, 1, 1, 1])
    return sc


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
    if True:
        for num in mynums:
            ds = s.load_athdf(num=num, output_id=2, load_method="yt")
            sc = make_sphere_volume(ds, num)
            foutdir = osp.join(os.fspath(s.basedir), "volume")
            os.makedirs(foutdir, exist_ok=True)
            fout = osp.join(foutdir, f"time_rotation_{num:04d}.png")
            sc.save(fout)

            n = gc.collect()
            print("Unreachable objects:", n, end=" ")
            print("Remaining Garbage:", end=" ")
            pprint.pprint(gc.garbage)
            sys.stdout.flush()

    # COMM.barrier()

    if COMM.rank == 0:
        fin = osp.join(os.fspath(s.basedir), "volume/time_rotation_*.png")
        fout = osp.join(os.fspath(s.basedir), "{0:s}_volume.mp4".format(s.basename))
        make_movie(fin, fout, fps_in=15, fps_out=15)
        from shutil import copyfile

        copyfile(
            fout,
            osp.join(
                "/tigress/changgoo/public_html/temporary_movies/SNTI",
                osp.basename(fout),
            ),
        )
