"""Correctness tests for the evaluation metrics, validated against skimage / elf references."""
import numpy as np
from skimage.metrics import variation_of_information as voi_ref

import bioimage_py as bp


# --- references -----------------------------------------------------------------------

def _elf_rand(segmentation, groundtruth):
    """elf's exact compute_rand_scores (the metric we port; skimage's adapted-rand differs)."""
    from collections import Counter
    seg, gt = segmentation.ravel(), groundtruth.ravel()
    n = seg.size
    a = Counter(gt.tolist())   # elf: a = groundtruth
    b = Counter(seg.tolist())
    pair = Counter(zip(gt.tolist(), seg.tolist()))
    sum_a = sum(c * c for c in a.values())
    sum_b = sum(c * c for c in b.values())
    sum_ab = sum(c * c for c in pair.values())
    prec, rec = sum_ab / sum_b, sum_ab / sum_a
    are = 1.0 - (2 * prec * rec) / (prec + rec)
    ri = 1.0 - (sum_a + sum_b - 2 * sum_ab) / (n * n)
    return are, ri


def _dice_ref(a, b, ts=0, tg=0):
    a = (a if ts is None else a > ts).astype("float64")
    b = (b if tg is None else b > tg).astype("float64")
    return 2 * np.sum(a * b) / (np.sum(a) + np.sum(b) + 1e-7)


def _best_dice_ref(a, b):
    """Brute-force best dice per a-object over b-objects (label 0 excluded, eps-free)."""
    a_labels = np.setdiff1d(np.unique(a), [0])
    b_labels = np.setdiff1d(np.unique(b), [0])
    if a_labels.size == 0 or b_labels.size == 0:
        return 0.0
    best = []
    for la in a_labels:
        am = a == la
        sa = am.sum()
        best.append(max(2 * np.sum(am & (b == lb)) / (sa + np.sum(b == lb)) for lb in b_labels))
    return float(np.mean(best))


def _sbd_ref(seg, gt):
    return min(_best_dice_ref(seg, gt), _best_dice_ref(gt, seg))


def _rng_pair(rng, shape=(64, 64), lo=1, hi=20):
    return (rng.integers(lo, hi, size=shape).astype("uint64"),
            rng.integers(lo, hi, size=shape).astype("uint64"))


_SEG = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype="uint64")


# --- variation of information ---------------------------------------------------------

def test_vi_against_skimage(rng):
    x, y = _rng_pair(rng)
    vis, vim = bp.evaluation.variation_of_information(x, y)
    skim = voi_ref(x, y)  # skimage returns [h(y|x), h(x|y)] = [merge, split]
    assert np.isclose(vis, skim[1])
    assert np.isclose(vim, skim[0])


def test_vi_identical():
    vis, vim = bp.evaluation.variation_of_information(_SEG, _SEG)
    assert np.isclose(vis, 0.0) and np.isclose(vim, 0.0)


def test_vi_pure_split_and_merge():
    gt_merge = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 4, 4], [3, 3, 4, 4]], dtype="uint64")
    vis, vim = bp.evaluation.variation_of_information(_SEG, gt_merge)  # seg splits gt object 1
    assert vis > 0.0 and np.isclose(vim, 0.0)
    vis, vim = bp.evaluation.variation_of_information(gt_merge, _SEG)  # seg merges -> only merge error
    assert np.isclose(vis, 0.0) and vim > 0.0


def test_vi_use_log2(rng):
    x, y = _rng_pair(rng, (32, 32), 1, 10)
    vis2, vim2 = bp.evaluation.variation_of_information(x, y, use_log2=True)
    vise, vime = bp.evaluation.variation_of_information(x, y, use_log2=False)
    assert np.isclose(vis2, vise / np.log(2)) and np.isclose(vim2, vime / np.log(2))


# --- object vi ------------------------------------------------------------------------

def test_object_vi_identical():
    df = bp.evaluation.object_vi(_SEG, _SEG)
    assert list(df.columns) == ["label", "vi_split", "vi_merge"]
    assert set(df["label"]) == {1, 2, 3, 4}
    assert np.allclose(df[["vi_split", "vi_merge"]].to_numpy(), 0.0)


def test_object_vi_split_object():
    gt = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 4, 4], [3, 3, 4, 4]], dtype="uint64")
    df = bp.evaluation.object_vi(_SEG, gt).set_index("label")
    assert df.loc[1, "vi_merge"] > 0.0       # gt object 1 is split in seg
    assert np.isclose(df.loc[3, "vi_merge"], 0.0)
    assert np.isclose(df.loc[4, "vi_merge"], 0.0)


# --- rand index -----------------------------------------------------------------------

def test_rand_against_elf_formula(rng):
    x, y = _rng_pair(rng)
    are, ri = bp.evaluation.rand_index(x, y)
    are_ref, ri_ref = _elf_rand(x, y)
    assert np.isclose(are, are_ref) and np.isclose(ri, ri_ref)


def test_rand_identical():
    are, ri = bp.evaluation.rand_index(_SEG, _SEG)
    assert np.isclose(are, 0.0) and np.isclose(ri, 1.0)


def test_rand_increasing_error():
    gt_merge = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 4, 4], [3, 3, 4, 4]], dtype="uint64")
    gt_worse = np.ones((4, 4), dtype="uint64")
    are_partial, _ = bp.evaluation.rand_index(_SEG, gt_merge)
    are_worse, _ = bp.evaluation.rand_index(_SEG, gt_worse)
    assert 0.0 < are_partial < are_worse


# --- cremi ----------------------------------------------------------------------------

def test_cremi_components(rng):
    x, y = _rng_pair(rng)
    vis, vim, are, cs = bp.evaluation.cremi_score(x, y)
    assert (vis, vim) == bp.evaluation.variation_of_information(x, y)
    assert np.isclose(are, bp.evaluation.rand_index(x, y)[0])
    assert np.isclose(cs, np.sqrt(are * (vis + vim)))


def test_cremi_identical():
    assert np.allclose(bp.evaluation.cremi_score(_SEG, _SEG), 0.0)


# --- matching -------------------------------------------------------------------------

def _check_all(scores, value):
    for key in ("precision", "recall", "segmentation_accuracy", "f1"):
        assert np.isclose(scores[key], value), f"{key}={scores[key]} != {value}"


def test_matching_example():
    scores = bp.evaluation.matching(np.array([0, 1, 2, 3, 4]), np.array([0, 1, 0, 0, 0]))
    assert scores == {"precision": 0.25, "recall": 1.0, "segmentation_accuracy": 0.25, "f1": 0.4}


def test_matching_identical(rng):
    x, _ = _rng_pair(rng, (128, 128), 0, 10)
    _check_all(bp.evaluation.matching(x, x), 1.0)


def test_matching_threshold():
    a, b = np.array([0, 0, 1, 1, 1, 2]), np.array([0, 0, 1, 1, 2, 2])
    _check_all(bp.evaluation.matching(a, b, threshold=0.7), 0.0)
    _check_all(bp.evaluation.matching(a, b, threshold=0.5), 1.0)


def test_matching_ignore_label():
    a, b = np.array([0, 1]), np.array([1, 2])
    scores = bp.evaluation.matching(a, b, ignore_label=0)
    assert np.isclose(scores["precision"], 1.0) and np.isclose(scores["recall"], 0.5)
    assert np.isclose(scores["f1"], 2.0 / 3.0)
    _check_all(bp.evaluation.matching(a, b, ignore_label=None), 1.0)


def test_mean_segmentation_accuracy(rng):
    x, y = _rng_pair(rng, (128, 128), 0, 10)
    assert bp.evaluation.mean_segmentation_accuracy(x, x) == 1.0
    score = bp.evaluation.mean_segmentation_accuracy(x, y)
    assert 0.0 <= score < 1.0
    mean, accs = bp.evaluation.mean_segmentation_accuracy(x, y, return_accuracies=True)
    assert np.isclose(mean, np.mean(accs))


# --- centroid / coordinate matching ---------------------------------------------------

def test_coordinate_matching_identical():
    pts = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 1.0]])
    _check_all(bp.evaluation.coordinate_matching(pts, pts, distance_threshold=0.5), 1.0)


def test_coordinate_matching_disjoint():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[100.0, 100.0], [200.0, 200.0]])
    _check_all(bp.evaluation.coordinate_matching(a, b, distance_threshold=1.0), 0.0)


def test_coordinate_matching_example():
    # 3 predicted points, 2 reference points; two predictions match within the threshold.
    pred = np.array([[0.0, 0.0], [10.0, 0.0], [50.0, 50.0]])
    gt = np.array([[0.0, 1.0], [10.0, 1.0]])
    scores = bp.evaluation.coordinate_matching(pred, gt, distance_threshold=2.0)
    assert np.isclose(scores["precision"], 2.0 / 3.0)   # tp=2, fp=1
    assert np.isclose(scores["recall"], 1.0)            # tp=2, fn=0
    assert np.isclose(scores["segmentation_accuracy"], 2.0 / 3.0)
    assert np.isclose(scores["f1"], 0.8)


def test_coordinate_matching_threshold():
    a, b = np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])  # Euclidean distance 5.
    _check_all(bp.evaluation.coordinate_matching(a, b, distance_threshold=4.0), 0.0)
    _check_all(bp.evaluation.coordinate_matching(a, b, distance_threshold=5.0), 1.0)


def test_coordinate_matching_resolution():
    a, b = np.array([[0.0, 0.0]]), np.array([[0.0, 3.0]])  # distance 3 in voxels.
    _check_all(bp.evaluation.coordinate_matching(a, b, distance_threshold=4.0), 1.0)
    # Anisotropic spacing doubles the axis-1 distance to 6, beyond the threshold.
    _check_all(bp.evaluation.coordinate_matching(a, b, distance_threshold=4.0, resolution=[1.0, 2.0]),
               0.0)


def test_coordinate_matching_empty():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    _check_all(bp.evaluation.coordinate_matching(np.empty((0, 2)), a, distance_threshold=1.0), 0.0)
    _check_all(bp.evaluation.coordinate_matching(a, np.empty((0, 2)), distance_threshold=1.0), 0.0)


def test_centroid_matching_identical():
    seg = np.array([[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 0], [3, 3, 0, 0]], dtype="uint64")
    _check_all(bp.evaluation.centroid_matching(seg, seg, distance_threshold=0.5), 1.0)


def test_centroid_matching_consistency(rng):
    # The high-level wrapper agrees with the coordinate form fed scipy's center-of-mass references.
    from scipy.ndimage import center_of_mass
    from skimage.measure import label as sklabel

    seg = sklabel(rng.random((64, 64)) > 0.7).astype("uint64")
    gt = sklabel(rng.random((64, 64)) > 0.7).astype("uint64")

    def coms(arr):
        labels = np.setdiff1d(np.unique(arr), [0])
        return np.array(center_of_mass(np.ones_like(arr), arr, labels), dtype="float64")

    ref_a, ref_b = coms(seg), coms(gt)
    for thr in (1.0, 3.0, 10.0):
        assert (bp.evaluation.centroid_matching(seg, gt, distance_threshold=thr)
                == bp.evaluation.coordinate_matching(ref_a, ref_b, distance_threshold=thr))


# --- dice -----------------------------------------------------------------------------

def test_dice_identical_and_disjoint():
    seg = (_SEG > 2).astype("uint8")
    assert np.isclose(bp.evaluation.dice_score(seg, seg), 1.0)
    a, b = np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1])
    assert np.isclose(bp.evaluation.dice_score(a, b), 0.0)


def test_dice_against_reference(rng):
    a = rng.integers(0, 5, size=(40, 40))
    b = rng.integers(0, 5, size=(40, 40))
    assert np.isclose(bp.evaluation.dice_score(a, b), _dice_ref(a, b))
    # soft dice on raw values (no thresholding)
    fa, fb = rng.random((40, 40)), rng.random((40, 40))
    assert np.isclose(bp.evaluation.dice_score(fa, fb, threshold_seg=None, threshold_gt=None),
                      _dice_ref(fa, fb, ts=None, tg=None))


# --- symmetric best dice --------------------------------------------------------------

def test_sbd_identical_and_reference(rng):
    x, y = _rng_pair(rng, (64, 64), 1, 30)
    assert np.isclose(bp.evaluation.symmetric_best_dice_score(x, x), 1.0)
    score = bp.evaluation.symmetric_best_dice_score(x, y)
    assert np.isclose(score, _sbd_ref(x, y))
    # symmetric in its arguments
    assert np.isclose(score, bp.evaluation.symmetric_best_dice_score(y, x))


# --- two-layer consistency & ignore ---------------------------------------------------

def test_two_layer_consistency(rng):
    x, y = _rng_pair(rng)
    ct = bp.evaluation.contingency_table(x, y)
    assert bp.evaluation.variation_of_information(x, y) == bp.evaluation.vi_scores(ct)
    assert bp.evaluation.rand_index(x, y) == bp.evaluation.rand_scores(ct)
    assert bp.evaluation.matching(x, y) == bp.evaluation.matching_scores(ct)


def test_ignore_seg_gt_single_sided(rng):
    # Single-sided ignore_gt == evaluating only where gt is not the ignore label.
    x, y = _rng_pair(rng, (40, 40), 0, 6)
    keep = y != 0
    vis_ref, vim_ref = bp.evaluation.variation_of_information(x[keep], y[keep])
    vis, vim = bp.evaluation.variation_of_information(x, y, ignore_gt=[0])
    assert np.isclose(vis, vis_ref) and np.isclose(vim, vim_ref)
