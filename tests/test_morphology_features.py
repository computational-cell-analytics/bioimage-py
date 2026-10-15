"""Correctness tests for the advanced per-object regionprops workflow."""
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label as sklabel
from skimage.measure import marching_cubes, mesh_surface_area, regionprops

import bioimage_py as bp


def _blobs(shape, rng):
    """A labeled volume of irregular blobs (smoothed noise -> threshold -> connected components)."""
    field = gaussian_filter(rng.random(shape), sigma=1.6)
    return sklabel(field > field.mean() + 0.8 * field.std()).astype("uint64")


def _direct_surface(mask, resolution):
    """Independent marching-cubes surface area on a padded mask (mirrors the op)."""
    if not mask.any():
        return 0.0
    padded = np.pad(mask, 1).astype("float32")
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=tuple(resolution))
        return float(mesh_surface_area(verts, faces))
    except (RuntimeError, ValueError):
        return 0.0


def test_regionprops_matches_skimage_3d(rng):
    seg = _blobs((28, 36, 40), rng)
    res = (2.0, 1.0, 1.5)
    table = bp.morphology.morphology(seg)
    df = bp.morphology.regionprops(seg, table, resolution=res, compute_surface=True,
                                   num_workers=2).set_index("label")

    props = {rp.label: rp for rp in regionprops(seg.astype("int32"), spacing=res)}
    assert set(df.index) == set(props)
    for label, rp in props.items():
        row = df.loc[label]
        n = int((seg == label).sum())
        assert int(row["n_voxels"]) == n
        # The numpy moment features reproduce the scikit-image definitions exactly.
        assert np.isclose(row["area"], rp.area)              # area == physical volume under spacing
        assert np.isclose(row["area"], n * float(np.prod(res)))
        # atol covers float noise on degenerate (e.g. one-voxel-thick) objects whose minor axis ~ 0.
        assert np.isclose(row["axis_major_length"], rp.axis_major_length, atol=1e-6)
        assert np.isclose(row["axis_minor_length"], rp.axis_minor_length, atol=1e-6)
        assert np.isclose(row["extent"], rp.extent)
        assert np.isclose(row["equivalent_diameter_area"], rp.equivalent_diameter_area)
        # surface area matches an independent marching-cubes on the same object
        assert np.isclose(row["surface_area"], _direct_surface((seg == label)[rp.slice], res))
        # the corrected centroid is always an actual object voxel
        vox = np.round(np.array([row[f"centroid_{a}"] for a in ("z", "y", "x")]) / np.array(res))
        assert bool((seg == label)[tuple(vox.astype(int))])


def test_centroid_correction_concave(rng):
    # A C-shaped object whose center-of-mass falls in the (background) hole.
    seg = np.zeros((6, 24, 24), dtype="uint64")
    seg[1:5, 4:20, 4:8] = 7      # left bar
    seg[1:5, 4:8, 4:20] = 7      # top bar
    seg[1:5, 16:20, 4:20] = 7    # bottom bar
    res = (3.0, 1.0, 1.0)
    table = bp.morphology.morphology(seg)
    row = bp.morphology.regionprops(seg, table, resolution=res).set_index("label").loc[7]

    bb = table.set_index("label").loc[7]
    origin = np.array([int(bb.bb_min_z), int(bb.bb_min_y), int(bb.bb_min_x)])
    crop = (seg == 7)[origin[0]:int(bb.bb_max_z), origin[1]:int(bb.bb_max_y), origin[2]:int(bb.bb_max_x)]
    # com lands in the hole -> correction kicks in; expect the deepest-interior voxel (EDT argmax).
    com_local = np.array([bb.com_z, bb.com_y, bb.com_x]) - origin
    assert not crop[tuple(np.round(com_local).astype(int))]
    dt = distance_transform_edt(crop, sampling=res)
    expected = (np.array(np.unravel_index(np.argmax(dt), crop.shape)) + origin) * np.array(res)
    got = np.array([row[f"centroid_{a}"] for a in ("z", "y", "x")])
    assert np.allclose(got, expected)


def test_centroid_kept_when_inside():
    # A solid box: the center-of-mass is interior, so it is kept (scaled to physical units).
    seg = np.zeros((10, 16, 18), dtype="uint64")
    seg[2:8, 3:13, 4:16] = 1
    res = (2.0, 1.0, 1.5)
    table = bp.morphology.morphology(seg)
    row = bp.morphology.regionprops(seg, table, resolution=res).set_index("label").loc[1]
    com = table.set_index("label").loc[1][["com_z", "com_y", "com_x"]].to_numpy(dtype="float64")
    got = np.array([row[f"centroid_{a}"] for a in ("z", "y", "x")])
    assert np.allclose(got, com * np.array(res))


def test_resolution_scaling():
    seg = np.zeros((16, 20, 24), dtype="uint64")
    seg[2:10, 3:15, 4:20] = 1
    table = bp.morphology.morphology(seg)
    n = int((seg == 1).sum())
    d1 = bp.morphology.regionprops(seg, table, resolution=(1, 1, 1)).iloc[0]
    d2 = bp.morphology.regionprops(seg, table, resolution=(2, 1, 1)).iloc[0]
    assert np.isclose(d1["area"], n)            # area == physical volume; voxel units here
    assert np.isclose(d2["area"], 2 * n)
    assert np.isclose(d2["centroid_z"], 2 * d1["centroid_z"])  # only the z axis is scaled


def test_table_from_csv_and_xlsx(tmp_path, rng):
    seg = _blobs((18, 24, 26), rng)
    res = (2.0, 1.0, 1.0)
    table = bp.morphology.morphology(seg)
    ref = bp.morphology.regionprops(seg, table, resolution=res)

    csv = tmp_path / "table.csv"
    table.to_csv(csv, index=False)
    pd.testing.assert_frame_equal(ref, bp.morphology.regionprops(seg, str(csv), resolution=res))

    xlsx = tmp_path / "table.xlsx"
    table.to_excel(xlsx, index=False)
    pd.testing.assert_frame_equal(ref, bp.morphology.regionprops(seg, str(xlsx), resolution=res))


def test_output_path_written(tmp_path, rng):
    seg = _blobs((16, 20, 22), rng)
    table = bp.morphology.morphology(seg)
    out = tmp_path / "features.csv"
    df = bp.morphology.regionprops(seg, table, output_path=str(out))
    assert out.exists()
    assert list(pd.read_csv(out)["label"]) == list(df["label"])


def test_regionprops_verbose(rng, capsys):
    seg = _blobs((16, 20, 22), rng)
    table = bp.morphology.morphology(seg)
    quiet = bp.morphology.regionprops(seg, table, verbose=False)
    assert "regionprops" not in capsys.readouterr().err
    loud = bp.morphology.regionprops(seg, table)
    assert "regionprops" in capsys.readouterr().err
    pd.testing.assert_frame_equal(quiet, loud)


def test_regionprops_empty():
    seg = np.zeros((6, 8, 10), dtype="uint64")
    table = bp.morphology.morphology(seg)
    df = bp.morphology.regionprops(seg, table, compute_surface=True)
    assert len(df) == 0
    assert {"label", "n_voxels", "area", "surface_area"} <= set(df.columns)


def test_pre_cleanup_forwarded_subprocess(zarr_factory, rng, tmp_path):
    """morphology()/regionprops() forward pre_cleanup so per-worker timings can be persisted.

    The callback fires on the orchestrating process with the job temp folder right before it is
    deleted; the distributed harness has already written one ``timings/<task_id>.json`` per task.
    """
    import glob
    import json
    import os

    from bioimage_py.runner.config import RunnerConfig

    seg = _blobs((24, 32, 36), rng)
    z = zarr_factory(seg, chunks=(12, 16, 18))
    captured = {}

    def save(name):
        def _cb(tmp_folder):
            recs = [json.load(open(p))
                    for p in glob.glob(os.path.join(tmp_folder, "timings", "*.json"))]
            captured[name] = recs
        return _cb

    cfg = RunnerConfig(tmp_root=str(tmp_path))
    table = bp.morphology.morphology(z, num_workers=2, block_shape=(12, 16, 18),
                                     job_type="subprocess", job_config=cfg,
                                     pre_cleanup=save("morphology"))
    df = bp.morphology.regionprops(z, table, resolution=(2.0, 1.0, 1.0), num_workers=2,
                                   job_type="subprocess", job_config=cfg,
                                   pre_cleanup=save("regionprops"))

    assert set(captured) == {"morphology", "regionprops"}
    for name, recs in captured.items():
        assert recs, f"{name}: no per-task timing records found"
        assert all("compute_s" in r for r in recs), f"{name}: timing record missing compute_s"
    assert len(df) == len(table) > 0  # the workflow still produced the expected objects


def test_regionprops_requires_integer():
    with pytest.raises(ValueError, match="integer"):
        bp.morphology.regionprops(np.zeros((5, 6, 7), dtype="float32"), pd.DataFrame({"label": [1]}))


def test_regionprops_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        bp.morphology.regionprops(np.zeros((5, 6, 7), dtype="uint64"), pd.DataFrame({"label": [1]}))
