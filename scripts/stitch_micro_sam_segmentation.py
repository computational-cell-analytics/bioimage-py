"""Segment a whole-slide image tile-by-tile and stitch the tiles.

Uses micro_sam's instance segmentation functionality over tiles and `bioimage_py.segmentation.stitch_segmentation`
to stitch the results.
"""
import argparse
from typing import Callable

import numpy as np
import bioimage_py as bp


def spatial_shape(image):
    """Return the spatial shape of ``image``, dropping a trailing RGB(A) channel axis if present."""
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return tuple(int(s) for s in image.shape[:2])
    return tuple(int(s) for s in image.shape)


def build_segmentation_function(model_type, device, verbose=False) -> Callable[[np.ndarray, int], np.ndarray]:
    """Build the per-tile segmentation function for `stitch_segmentation`.

    The SAM model is loaded once and captured in the closure (the load is expensive, so it must not
    happen per tile). This keeps the function local-execution only: a loaded torch model cannot be
    cloudpickled for the distributed backends, which would instead load it lazily inside each worker.

    Args:
        model_type: The micro-sam model, e.g. ``"vit_b_lm"`` or ``"vit_b_histopathology"``.
        device: The torch device (``None`` lets micro-sam auto-select GPU/CPU).
        verbose: Whether micro-sam prints per-tile progress.

    Returns:
        A function ``f(tile_input, tile_id) -> labels`` returning a ``uint32`` label image of the
        tile's (haloed) spatial shape, as required by `stitch_segmentation`.
    """
    from micro_sam.automatic_segmentation import automatic_instance_segmentation
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter

    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, device=device)

    def segment(tile_input: np.ndarray, tile_id: int) -> np.ndarray:
        # ndim=2 forces 2d segmentation even for RGB tiles (HxWx3); no tile_shape/halo here, so
        # micro-sam segments this single tile rather than re-tiling it. Background is 0.
        segmentation = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter, input_path=tile_input, ndim=2, verbose=verbose,
        )
        return np.asarray(segmentation).astype("uint32")

    return segment


def main() -> None:
    """Fetch the whole-slide example, segment it tile-wise with micro-sam, and stitch the tiles."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop", type=int, default=None,
                        help="Crop the image to a CROPxCROP top-left region (fast end-to-end test).")
    args = parser.parse_args()

    import imageio.v3 as imageio
    from micro_sam.sample_data import fetch_wholeslide_example_data

    data_dir = "./data"
    tile_shape = (512, 512)
    tile_overlap = (64, 64)

    # The number of parallel workers and the job type.
    # Set the job type to "slurm" to run on HPC with slurm instead of locally.
    num_workers = 8
    job_type = "local"

    image_path = fetch_wholeslide_example_data(data_dir)
    image = np.asarray(imageio.imread(image_path))
    if args.crop is not None:
        image = image[:args.crop, :args.crop]

    shape = spatial_shape(image)
    n_tiles = int(np.prod([-(-s // t) for s, t in zip(shape, tile_shape)]))  # ceil-div per axis
    print(f"Image shape {image.shape} (spatial {shape}); tiling into ~{n_tiles} "
          f"tile(s) of {tile_shape} with overlap {tile_overlap}.")

    segmentation_function = build_segmentation_function(model_type="vit_b_lm", device=None, verbose=False)
    segmentation = bp.segmentation.stitch_segmentation(
        image, segmentation_function, tile_shape, tile_overlap,
        shape=shape, num_workers=num_workers, job_type=job_type,
    )
    segmentation = np.asarray(segmentation)

    import napari
    viewer = napari.Viewer()
    viewer.add_image(image, name="image")
    viewer.add_labels(segmentation.astype("uint32"), name="stitched-segmentation")
    napari.run()


if __name__ == "__main__":
    main()
