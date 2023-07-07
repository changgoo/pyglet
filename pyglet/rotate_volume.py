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

import yt


def make_sphere_volume(ds, num, radius=96):
    from yt.visualization.volume_rendering.api import Scene, create_volume_source

    n1 = 400
    n2 = 500
    zoomfactor = (
        2.0
        if num < n1
        else 1.0
        if num > n2
        else (1 * (num - n1) / (n2 - n1) + 2 * (num - n2) / (n1 - n2))
    )

    sp = ds.sphere(
        center=ds.domain_center, radius=(radius / max(zoomfactor - 0.5, 1), "pc")
    )
    sc = yt.create_scene(sp)

    source = sc[0]
    source.set_field("density")
    source.set_log(True)
    source.set_use_ghost_zones(True)

    bounds = (1, 1.0e2)
    tf = yt.ColorTransferFunction((np.log10(bounds[0]), np.log10(bounds[1])))
    tf.add_layers(5, w=0.01, alpha=np.logspace(0, 3, 5), colormap="winter")

    source.set_transfer_function(tf)
    source.tfh.plot("transfer_function_density.png")

    # source.set_transfer_function(tf)
    # source.tfh.plot("transfer_function.png", profile_field="density")

    # Add temperature

    field = "temperature"

    vol2 = create_volume_source(sp, field=field)
    vol2.set_use_ghost_zones(True)

    bounds = (1.0e5, 1.0e6)
    tf = yt.ColorTransferFunction((np.log10(bounds[0]), np.log10(bounds[1])))
    tf.clear()
    tf.add_layers(3, 0.02, alpha=[100] * 3, colormap="autumn")

    vol2.set_transfer_function(tf)
    vol2.tfh.plot("transfer_function_temperature.png")
    sc.add_source(vol2)

    cam = sc.camera
    cam.zoom(zoomfactor)
    cam.resolution = (1024, 1024)

    sc.annotate_domain(ds, color=[1, 1, 1, 1])
    return sc


if __name__ == "__main__":
    yt.enable_parallelism()

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
    foutdir = osp.join(os.fspath(s.basedir), "volume")
    os.makedirs(foutdir, exist_ok=True)

    ds = s.load_athdf(num=320, output_id=2, load_method="yt")
    sc = make_sphere_volume(ds, 320)

    # rotate iclicks
    Nrot = 512
    frame = 0
    for _ in sc.camera.iter_rotate(2 * np.pi, Nrot):
        frame += 1
        fout = osp.join(foutdir, f"rotation_{frame:04d}.png")
        sc.save(fout)
        sys.stdout.flush()

    n = gc.collect()
    print("Unreachable objects:", n, end=" ")
    print("Remaining Garbage:", end=" ")
    pprint.pprint(gc.garbage)
    sys.stdout.flush()
