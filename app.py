from functools import reduce

import math
import os
import uuid
from enum import Enum
from typing import List, Optional
import numpy as np
import xarray as xr
from colorthief import ColorThief
from matplotlib import cm
from PIL import Image
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mean
import cv2
from mikro_next.api.schema import (
    Image,
    Era,
    create_era,
    images,
    create_roi,
    create_stage,
    ImageFilter,
    Table,
    from_parquet_like,
    Snapshot,
    File,
    RoiKind,
    Dataset,
    AffineTransformationView,
    FiveDVector,
    FourByFourMatrix,
    from_array_like,
    PartialROIViewInput,
    PartialScaleViewInput,
    PartialDerivedViewInput ,
    ROI,
    Stage,
    get_image,
)
import operator
from arkitekt_next import register, log, model
from functools import partial
from skimage import transform
import datetime
from typing import Tuple, Generator
from skimage import data
import skimage
from scipy import ndimage
from sklearn.cluster import DBSCAN
import pandas as pd
from dataclasses import dataclass

@model 
@dataclass
class Position:
    x: float
    y: float
    z: float


@model 
@dataclass
class TwoDSize:
    x: float
    y: float








class Colormap(Enum):
    VIRIDIS = partial(cm.viridis)  # partial needed to make it register as an enum value
    PLASMA = partial(cm.plasma)


@register(collections=["quantitative"])
def measure_max(
    rep: Image,
) -> float:
    """Measure Max

    Measures the maxium value of an image

    Args:
        rep (Image): The image
        key (str, optional): The key to use for the metric. Defaults to "max".

    Returns:
        Representation: The Back
    """
    return float(rep.data.max().compute())


@register(collections=["creation"])
def create_era_func(
    name: str = "max",
) -> Era:
    """Create Era Now

    Creates an era with the current time as a starttime

    Returns:
        Representation: The Back
    """
    return create_era(name=name, begin=datetime.datetime.now())


@register(collections=["collection"])
def iterate_images(
    dataset: Dataset,
) -> Generator[Image, None, None]:
    """Iterate Images

    Iterate over all images in a dataset

    Args:
        rep (Dataset): The dataset

    yields:
        Representation: The image
    """
    for x in images(filter=ImageFilter(dataset=dataset)):
        yield x


@register(collections=["quantitative"])
def measure_sum(
    rep: Image,
) -> float:
    """Measure Sum

    Measures the sum of all values of an image

    Args:
        rep (Image): The image

    Returns:
        Representation: The Back
    """
    return float(rep.data.sum().compute())


@register(collections=["quantitative"])
def measure_fraction(
    rep: Image,
    value: float = 1,
) -> float:
    """Measure Fraction

    Measures the appearance of this value in the image (0-1)

    Args:
        rep (OmeroFiRepresentationFragmentle): The image.

    Returns:
        Representation: The Back
    """
    x = rep.data == value
    sum = x.sum().compute()
    all_values = reduce(lambda x, t: x * t, rep.data.shape, 1)

    return float(sum / all_values)


@register(collections=["quantitative"])
def measure_basics(
    rep: Image,
) -> Tuple[float, float, float]:
    """Measure Basic Metrics

    Measures basic meffffftrics of an image like max, mifffffn, mean

    Args:
        rep (OmeroFiRepresentationFragmentle): The image

    Returns:
        float: The max
        float: The mean
        float: The min
    """

    x = rep.data.compute()

    return float(x.max()), float(x.mean()), float(x.min())

@register(collections=["conversion"])
def t_to_frame(
    rep: Image,
    interval: int = 1,
    key: str = "frame",
) -> Generator[ROI, None, None]:
    """T to Frame

    Converts a time series to a single frame

    Args:
        rep (RepresentationFragment): The Representation
        frame (int): The frame to select

    Returns:
        RepresentationFragment: The new Representation
    """
    assert "t" in rep.data.dims, "Cannot convert non time series to frame"

    for i in range(rep.data.sizes["t"]):
        if i % interval == 0:
            yield create_roi(
                representation=rep,
                label=f"{key} {i}",
                type=RoiKind.FRAME,
                tags=[f"t{i}", "frame"],
                vectors=[FiveDVector(t=i), FiveDVector(t=i + interval)],
            )


@register(collections=["conversion"])
def z_to_slice(
    rep: Image,
    interval: int = 1,
    key: str = "Slice",
) -> Generator[ROI, None, None]:
    """Z to Slice

    Creates a slice roi for each z slice

    Args:
        rep (RepresentationFragment): The Representation
        frame (int): The frame to select

    Returns:
        RepresentationFragment: The new Representation
    """
    assert "z" in rep.data.dims, "Cannot convert non time series to frame"

    for i in range(rep.data.sizes["z"]):
        if i % interval == 0:
            yield create_roi(
                representation=rep,
                label=f"{key} {i}",
                type=RoiKind.SLICE,
                tags=[f"z{i}", "frame"],
                vectors=[FiveDVector(z=i), FiveDVector(z=i + interval)],
            )


def cropND(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


@register(collections=["conversion", "cropping"])
def crop_image(
    roi: ROI,
) -> Image:
    """Crop Image

    Crops an Image based on a ROI

    Args:
        roi (ROI): The ROI to crop

    Returns:
        Representation: The Back
    """
    if rep is None:
        rep = get_image(roi.representation.id)

    array = rep.data
    if roi.type == RoiKind.RECTANGLE:
        x_start = roi.vectors[0].x
        y_start = roi.vectors[0].y
        x_end = roi.vectors[0].x
        y_end = roi.vectors[0].y

        for vector in roi.vectors:
            if vector.x < x_start:
                x_start = vector.x
            if vector.x > x_end:
                x_end = vector.x
            if vector.y < y_start:
                y_start = vector.y
            if vector.y > y_end:
                y_end = vector.y

        roi.vectors[0]

        array = array.sel(
            x=slice(math.floor(x_start), math.floor(x_end)),
            y=slice(math.floor(y_start), math.floor(y_end)),
        )

        return from_array_like(
            array,
            name="Cropped " + rep.name,
            roi_views=[PartialROIViewInput(roi=roi)],
        )

    if roi.type == RoiKind.FRAME:
        array = array.sel(
            t=slice(math.floor(roi.vectors[0].t), math.floor(roi.vectors[1].t))
        )

        return from_array_like(
            array,
            name="Cropped " + rep.name,
            roi_views=[PartialROIViewInput(roi=roi)],
        )

    if roi.type == RoiKind.SLICE:
        array = array.sel(
            z=slice(math.floor(roi.vectors[0].z), math.floor(roi.vectors[1].z))
        )

        return from_array_like(
            array,
            name="Cropped " + rep.name,
            roi_views=[PartialROIViewInput(roi=roi)],
        )

    raise Exception(f"Roi Type {roi.type} not supported")


class DownScaleMethod(Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"


@register(collections=["processing", "scaling"])
def downscale_image(
    image: Image,
    factor: int = 2,
    depth=0,
    method: DownScaleMethod = DownScaleMethod.MEAN,
) -> Image:
    """Downscale

    Scales down the Representatoi by the factor of the provided

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        RepresentationFragment: The Downscaled image
    """
    s = tuple([1 if c == 1 else factor for c in image.data.squeeze().shape])

    newrep = multiscale(image.data.squeeze(), windowed_mean, s)

    return from_array_like(
            newrep,
            name="Scaled " + image.name,
            scale_views=[PartialScaleViewInput(scaleX=s, scaleY=s, scaleZ=s)],
        )


@register(collections=["processing", "scaling"])
def rescale(
    image: Image,
    factor_x: float = 2.0,
    factor_y: float = 2.0,
    factor_z: float = 2.0,
    factor_t: float = 1.0,
    factor_c: float = 1.0,
    anti_alias: bool = True,
    method: DownScaleMethod = DownScaleMethod.MEAN,
) -> Image:
    """Rescale

    Rescale the dimensions by the factors provided

    Args:
        rep (RepresentationFragment): The Image we should rescale

    Returns:
        RepresentationFragment: The Rescaled image
    """

    scale_map = {
        "x": factor_x,
        "y": factor_y,
        "z": factor_z,
        "t": factor_t,
        "c": factor_c,
    }

    squeezed_data = image.data.squeeze()
    dims = squeezed_data.dims

    s = tuple([scale_map[d] for d in dims])

    newrep = transform.rescale(squeezed_data.data, s, anti_aliasing=anti_alias)

    return from_array_like(
            xr.DataArray(newrep, dims=dims),
            name="Scaled " + image.name,
            scale_views=[PartialScaleViewInput(scaleX=scale_map["x"], scaleY=scale_map["y"], scaleZ=scale_map["z"])],
        )


@register(collections=["processing", "scaling"])
def resize(
    image: Image,
    dim_x: Optional[int],
    dim_y: Optional[int],
    dim_z: Optional[int],
    dim_t: Optional[int],
    dim_c: Optional[int],
    anti_alias: bool = True,
) -> Image:
    """Resize

    Resize the image to the dimensions provided

    Args:
        rep (RepresentationFragment): The Image we should resized

    Returns:
        RepresentationFragment: The resized image
    """

    scale_map = {
        "x": dim_x or image.data.sizes["x"],
        "y": dim_y or image.data.sizes["y"],
        "z": dim_z or image.data.sizes["z"],
        "t": dim_t or image.data.sizes["t"],
        "c": dim_c or image.data.sizes["c"],
    }

    squeezed_data = image.data.squeeze()
    dims = squeezed_data.dims

    s = tuple([scale_map[d] for d in dims])

    newrep = transform.resize(
        squeezed_data.data, s, anti_aliasing=anti_alias, preserve_range=True
    )

    newrep = skimage.util.img_as_uint(newrep)
    return from_array_like(
            xr.DataArray(newrep, dims=dims),
            name="Scaled " + image.name,
            scale_views=[PartialScaleViewInput(scaleX=scale_map["x"], scaleY=scale_map["y"], scaleZ=scale_map["z"])],
        )


class CropMethod(Enum):
    CENTER = "mean"
    TOP_LEFT = "top-left"
    BOTTOM_RIGHT = "bottom-right"


class ExpandMethod(Enum):
    PAD_ZEROS = "zeros"


@register(
    collections=["processing", "scaling"],
)
def resize_to_physical(
    image: Image,
    rescale_x: Optional[float],
    rescale_y: Optional[float],
    rescale_z: Optional[float],
    ensure_dim_x: Optional[int],
    ensure_dim_y: Optional[int],
    ensure_dim_z: Optional[int],
    crop_method: CropMethod = CropMethod.CENTER,
    pad_method: ExpandMethod = ExpandMethod.PAD_ZEROS,
    anti_alias: bool = True,
) -> Image:
    """Resize to Physical

    Resize the image to match the physical size of the dimensions,
    if the physical size is not provided, it will be assumed to be 1.

    Additional dimensions will be cropped or padded according to the
    crop_method and pad_method if the ensure_dim is provided

    Args:
        rep (RepresentationFragment): The Image we should resized
        rescale_x (Optional[float]): The physical size of the x dimension
        rescale_y (Optional[float]): The physical size of the y dimension
        rescale_z (Optional[float]): The physical size of the z dimension
        ensure_dim_x (Optional[int]): The size of the x dimension
        ensure_dim_y (Optional[int]): The size of the y dimension
        ensure_dim_z (Optional[int]): The size of the z dimension
        crop_method (CropMethod, optional): The method to crop the image. Defaults to crop center.
        pad_method (ExpandMethod, optional): The method to pad the image. Defaults to expand with zeros.

    Returns:
        RepresentationFragment: The resized image
    """

    affine_transformation: Optional[FourByFourMatrix] = None

    for view in image.views:
        if isinstance(view, AffineTransformationView):
            affine_transformation = view.affine_matrix


    if not affine_transformation:
        raise ValueError("No affine transformation found")

    # Extract the scale
    originial_scale = affine_transformation[0, 0], affine_transformation[1, 1], affine_transformation[2, 2]
    scale_map = {
        "x": rescale_x / originial_scale[0] if rescale_x else 1,
        "y": rescale_y / originial_scale[1] if rescale_y else 1,
        "z": rescale_z / originial_scale[2] if rescale_z else 1,
        "t": 1,
        "c": 1,
    }

    squeezed_data = image.data.squeeze()
    dims = squeezed_data.dims

    s = tuple([scale_map[d] for d in dims])

    if not all([d == 1 for d in s]):
        newrep = transform.rescale(
            squeezed_data.data.compute(), s, anti_aliasing=anti_alias, preserve_range=True
        )
    else:
        newrep = squeezed_data.data.compute()


    new_array = xr.DataArray(newrep, dims=dims)

    if ensure_dim_x or ensure_dim_y or ensure_dim_z:
        print(newrep.shape)

        size_map = {
            "x": ensure_dim_x or newrep.shape[2],
            "y": ensure_dim_y or newrep.shape[1],
            "z": ensure_dim_z or newrep.shape[0],
            "t": image.data.sizes["t"],
            "c": image.data.sizes["c"],
        }

        s = tuple([size_map[d] for d in dims])
        new_array = cropND(
            new_array,
            s,
        )

    return from_array_like(
        new_array,
        name="Scaled " + image.name,
        roi_views=[PartialScaleViewInput(scaleX=scale_map["x"], scaleY=scale_map["y"], scaleZ=scale_map["z"])],
    )


class ThresholdMethod(Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


@register(collections=["processing", "thresholding"])
def threshold_image(
    image: Image,
    threshold: float = 0.5,
    method: ThresholdMethod = ThresholdMethod.MEAN,
) -> Image:
    """Binarize

    Binarizes the image based on a threshold

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        RepresentationFragment: The Downscaled image
    """
    print(method)
    if method == ThresholdMethod.MEAN.value:
        m = image.data.mean()
    if method == ThresholdMethod.MAX.value:
        m = image.data.max()
    if method == ThresholdMethod.MIN.value:
        m = image.data.min()

    newrep = image.data > threshold * m

    return from_array_like(
        newrep,
        name=f"Thresholded {image.name}",
        derived_views=[PartialDerivedViewInput(originImage=image)],
    )


@register(collections=["processing", "projection"])
def maximum_intensity_projection(
    image: Image,
) -> Image:
    """Maximum Intensity Projection

    Projects the image onto the maximum intensity along the z axis

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        RepresentationFragment: The Downscaled image
    """
    m = image.data.max(dim="z").compute()

    print("Maximum?")
    print(m.max())
    print(m.min())

    return from_array_like(
        m,
        name=f"Thresholded {image.name}",
        derived_views=[PartialDerivedViewInput(originImage=image)],
    )


class CV2NormTypes(Enum):
    NORM_INF = cv2.NORM_INF
    NORM_L1 = cv2.NORM_L1
    NORM_L2 = cv2.NORM_L2
    NORM_MINMAX = cv2.NORM_MINMAX
    NORM_RELATIVE = cv2.NORM_RELATIVE
    NORM_TYPE_MASK = cv2.NORM_TYPE_MASK


@register(collections=["processing", "thresholding", "adaptive"])
def adaptive_threshold_image(
    image: Image,
    normtype: CV2NormTypes = CV2NormTypes.NORM_MINMAX,
) -> Image:
    """Adaptive Binarize

    Binarizes the image based on a threshold

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        RepresentationFragment: The Downscaled image
    """
    x = image.data.compute()

    thresholded = xr.DataArray(np.zeros_like(x), dims=x.dims, coords=x.coords)

    for c in range(x.sizes["c"]):
        for z in range(x.sizes["z"]):
            for t in range(x.sizes["t"]):
                img = x.sel(c=c, z=z, t=t)
                normed = cv2.normalize(img.data, None, 0, 255, normtype, cv2.CV_8U)
                thresholded[c, t, z, :, :] = cv2.adaptiveThreshold(
                    normed,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                )

    return from_array_like(
        thresholded,
        name=f"Thresholded {image.name}",
        derived_views=[PartialDerivedViewInput(originImage=image)],
    )


class CV2NormTypes(Enum):
    NORM_INF = cv2.NORM_INF
    NORM_L1 = cv2.NORM_L1
    NORM_L2 = cv2.NORM_L2
    NORM_MINMAX = cv2.NORM_MINMAX
    NORM_RELATIVE = cv2.NORM_RELATIVE
    NORM_TYPE_MASK = cv2.NORM_TYPE_MASK


@register(collections=["processing", "thresholding", "adaptive"])
def otsu_thresholding(
    image: Image,
    gaussian_blur: bool = False,
) -> Image:
    """Otsu Thresholding

    Binarizes the image based on a threshold, using Otsu's method
    with option to guassian blur the image before, images are normalized
    for each x,y slice before thresholding.

    Args:
        rep (RepresentationFragment): The Image to be thresholded
        gaussian_blur (bool): Whether to apply a gaussian blur before thresholding (kernel_size=5)

    Returns:
        RepresentationFragment: The thresholded image
    """
    x = image.data.compute()

    thresholded = xr.DataArray(np.zeros_like(x, dtype=np.uint8), dims=x.dims, coords=x.coords)
    print("Hallo")
    print(x.min())
    print(x.max())

    for c in range(x.sizes["c"]):
        for z in range(x.sizes["z"]):
            for t in range(x.sizes["t"]):
                img = x.sel(c=c, z=z, t=t).data
                 # Find min and max values
                min_val = np.min(img)
                max_val = np.max(img)
                
                # Apply min-max normalization
                normalized_arr = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                if gaussian_blur:
                    normalized_arr = cv2.GaussianBlur(img, (5, 5), 0)
                print(img)
                threshold, image = cv2.threshold(
                    normalized_arr,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                print(image, threshold)
                print(image.dtype)
                thresholded[c, t, z, :, :] = image

    print(thresholded.dtype)

    return from_array_like(
        thresholded,
        name=f"Thresholded {image.name}",
        derived_views=[PartialDerivedViewInput(originImage=image)],
    )






@register(collections=["conversion"])
def roi_to_position(
    roi: ROI,
    walk: bool = True,
) -> Position:
    """Roi to Position

    Transforms a ROI into a Position on the governing stage

    Args:
        roi (ROI): Walk the tree to find the origin
        walk (bool): Whether to walk the tree to find the origin

    Returns:
        Position: The position
    """

    smart_image = get_image(roi.image)

    while smart_image.omero is None or smart_image.omero.positions is None:
        smart_image = get_image(smart_image.origins[0])
        assert (
            smart_image.shape == roi.representation.shape
        ), "Could not find a matching position is not in the same space as the original (probably through a downsampling, or cropping)"

    omero = smart_image.omero
    affine_transformation = omero.affine_transformation
    shape = smart_image.shape
    position = smart_image.omero.positions[0]

    # calculate offset between center of roi and center of image
    print(position)
    print(roi.get_vector_data(dims="ctzyx"))
    center = roi.center_as_array()
    print(center)

    image_center = np.array(shape) / 2
    print(image_center[2:])
    print(center[2:])
    offsetz, offsety, offsetx = image_center[2:]
    z, y, x = center[2:]

    x_from_center = x - offsetx
    y_from_center = y - offsety
    z_from_center = z - offsetz

    # TODO: check if this is correct and extend to 3d
    vec_center = np.array([x_from_center, y_from_center, z_from_center])
    vec = np.matmul(np.array(affine_transformation).reshape((3, 3)), vec_center)
    new_pos_x, new_pos_y, new_pos_z = (
        np.array([position.x, position.y, position.z]) + vec
    )

    print(vec)

    print("Affine", affine_transformation)

    return Position(x=new_pos_x, y=new_pos_y, z=new_pos_z)


@register(collections=["conversion"])
def roi_to_physical_dimensions(
    roi: ROI,
) -> TwoDSize:
    """Rectangular Roi to Dimensions

    Measures the size of a Rectangular Roi in microns
    (only works for Rectangular ROIS)

    Parameters
    ----------
    roi : ROIFragment
        The roi to measure

    Returns
    -------
    height: float
        The height of the ROI in microns
    width: float
        The width of the ROI in microns
    """
    assert roi.type == RoiKind.RECTANGLE, "Only works for rectangular ROIs"
    smart_image = get_image(roi.image)

    while smart_image.omero is None or smart_image.omero.physical_size is None:
        smart_image = get_image(smart_image.origins[0])
        assert (
            smart_image.shape == roi.representation.shape
        ), "Could not find a matching position is not in the same space as the original (probably through a downsampling, or cropping)"

    physical_size = smart_image.omero.physical_size

    # Convert to a numpy array for easier manipulation
    points = roi.get_vector_data(dims="yx")

    # Find the minimum and maximum x and y coordinates
    min_y, min_x = np.min(points, axis=0)
    max_y, max_x = np.max(points, axis=0)

    # Calculate the width and height
    width = max_x - min_x
    height = max_y - min_y

    return TwoDSize(x=width * physical_size.x, y=height * physical_size.y)


@register(collections=["conversion"])
def rois_to_positions(
    roi: List[ROI],
) -> List[Position]:
    """Rois to Positions

    Transforms a List of Rois into a List of Positions on the governing stage

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        List[PositionFragment]: The Downscaled image
    """
    positions = []
    for r in roi:
        positions.append(roi_to_position(r))

    return positions


@register(collections=["creation"])
def create_stage_from_name(
    name: str,
) -> Stage:
    """Create New Stage

    Creates a new stage with the given name

    """

    return create_stage(name=name)



@register(collections=["collection"])
def get_files_ff(
    dataset: Dataset,
) -> List[File]:
    """Get all Omerfiles in Dataset

    Gets the files in an dataset at the time of the request
    """
    print(dataset)

    return [file for file in dataset.omerofiles if file is not None]


class DataKind(Enum):
    ASTRONAUT = partial(
        data.astronaut
    )  # partial needed to make it register as an enum value
    BRAIN = partial(data.brain)
    BRICK = partial(data.brick)
    CAMERA = partial(data.camera)
    CAT = partial(data.cat)
    CELL = partial(data.cell)
    CELLS_3D = partial(data.cells3d)
    CHECKERBOARD = partial(data.checkerboard)
    CHELSEA = partial(data.chelsea)
    CLOCK = partial(data.clock)
    COFFEE = partial(data.coffee)
    COINS = partial(data.coins)
    COLORWHEEL = partial(data.colorwheel)
    EAGLE = partial(data.eagle)
    GRASS = partial(data.grass)
    GRAVEL = partial(data.gravel)
    HORSE = partial(data.horse)
    HUBBLE_DEEP_FIELD = partial(data.hubble_deep_field)
    HUMAN_MITOSIS = partial(data.human_mitosis)
    IMMUNOHISTOCHEMISTRY = partial(data.immunohistochemistry)
    KIDNEY = partial(data.kidney)
    LILY = partial(data.lily)
    LOGO = partial(data.logo)
    MICROANEURYSMS = partial(data.microaneurysms)
    MOON = partial(data.moon)
    NICKEL_SOLIDIFICATION = partial(data.nickel_solidification)
    PAGE = partial(data.page)
    PROTEIN_TRANSPORT = partial(data.protein_transport)
    RETINA = partial(data.retina)
    ROCKET = partial(data.rocket)
    SHEPP_LOGAN_PHANTOM = partial(data.shepp_logan_phantom)
    SKIN = partial(data.skin)
    TEXT = partial(data.text)
    VORTEX = partial(data.vortex)

TWOD_DATA = set(
    [
        DataKind.BRICK,
        DataKind.CAMERA,
        DataKind.CELL,
        DataKind.CLOCK,
        DataKind.COINS,
        DataKind.EAGLE,
        DataKind.GRASS,
        DataKind.GRAVEL,
        DataKind.CHECKERBOARD,
        DataKind.HORSE,
        DataKind.HUMAN_MITOSIS,
        DataKind.MICROANEURYSMS,
        DataKind.MOON,
        DataKind.PAGE,
        DataKind.SHEPP_LOGAN_PHANTOM,
        DataKind.TEXT,
        DataKind.VORTEX,

    ]
)
THREED_DATA = set(
    [
        DataKind.BRAIN,
    ]
)
TWOD_D_RGB_DATA = set(
    [
        DataKind.ASTRONAUT,
        DataKind.CAT,
        DataKind.CHELSEA,
        DataKind.COFFEE,
        DataKind.COLORWHEEL,
        DataKind.HUBBLE_DEEP_FIELD,
        DataKind.IMMUNOHISTOCHEMISTRY,
        DataKind.RETINA,
        DataKind.ROCKET,
        DataKind.SKIN,
    ]
)
THREED_CHANNEL_DATA = set([DataKind.CELLS_3D])
ZXYC_DATA = set([DataKind.KIDNEY])
XYC_DATA = set([DataKind.LILY, DataKind.LOGO])
TXY_DATA = set([DataKind.NICKEL_SOLIDIFICATION])
TCXY_DATA = set([DataKind.PROTEIN_TRANSPORT])


@register(collections=["generator"])
def generate_test_image(
    kind: DataKind, attach_meta: bool = True
) -> Image:
    """Generate Test Image"""

    data = kind.value()
    print(data.shape)

    if kind in TWOD_DATA:
        data = xr.DataArray(data, dims=["y", "x"])
    elif kind in THREED_DATA:
        data = xr.DataArray(data, dims=["z", "y", "x"])
    elif kind in TWOD_D_RGB_DATA:
        data = xr.DataArray(data, dims=["y", "x", "c"])
    elif kind in THREED_CHANNEL_DATA:
        data = xr.DataArray(data, dims=["z", "c", "y", "x"])
       
    elif kind in ZXYC_DATA:
        data = xr.DataArray(data, dims=["z", "x", "y", "c"])
        
    elif kind in XYC_DATA:
        data = xr.DataArray(data, dims=["x", "y", "c"])
        
    elif kind in TXY_DATA:
        data = xr.DataArray(data, dims=["t", "x", "y"])
    elif kind in TCXY_DATA:
        data = xr.DataArray(data, dims=["t", "c", "x", "y"])
    else:
        raise Exception("Unknown Data Kind")


    

    return from_array_like(
        data,
        name=f"Test Image {kind.name}",
    )


@register(collections=["segmentation"])
def mark_clusters_of_size_rectangle(
    rep: Image,
    distance: float,
    min_size: int,
    max_size: Optional[int],
    c: Optional[int] = 0,
    t: Optional[int] = 0,
    z: Optional[int] = 0,
    limit: Optional[int] = None,
) -> List[ROI]:
    """Mark Clusters

    Takes a masked image and marks rois based on dense clusters of a certain size 
    and density of their center of mass

    Args:
        rep (RepresentationFragment): The image to label outliers for
        distance (float): The distance between points in a cluster (eps in DBSCAN)
        min_size (int): The minimum size of a cluster (min_samples in DBSCAN)
        max_size (Optional[int]): The maximum size of a cluster (threshold for number of labels in a cluster)
        c (Optional[int], optional): The channel to use. Defaults to 0.
        t (Optional[int], optional): The timepoint to use. Defaults to 0.
        z (Optional[int], optional): The z-slice to use. Defaults to 0.
        limit (Optional[int], optional): The maximum number of clusters to return. Defaults to None.


    Returns:
        List[ROIFragment]: The rois for the clusters
    """

    x = rep.data.sel(c=c, t=t, z=z).compute().data

    # %%
    centroids = ndimage.center_of_mass(x, x, range(1, x.max() + 1))
    centroids = np.array(centroids)

    dbscan = DBSCAN(eps=distance, min_samples=min_size).fit(centroids)
    dblabels = dbscan.labels_
    n_clusters_ = len(set(dblabels)) - (1 if -1 in dblabels else 0)
    n_noise_ = list(dblabels).count(-1)

    log(f"Estimated number of clusters: {n_clusters_}")
    log(f"Estimated number of noise points:  {n_noise_}")

    if limit is not None:
        if n_clusters_ > limit:
            log(f"Limiting number of clusters to {limit}")
            n_clusters_ = limit
    rois = []

    for cluster in range(0, n_clusters_):
        centrois = centroids[dblabels == cluster]
        if max_size is not None and len(centrois) > max_size:
            continue


        in_labels = []

        for centoid in centrois:
            label_at_centroid = x[int(centoid[0]), int(centoid[1])]
            if label_at_centroid != 0:
                in_labels.append(label_at_centroid)

        if len(in_labels) == 0:
            continue


        mask = np.isin(x, in_labels)
        y_coords, x_coords = np.where(mask)

        y1 = np.min(y_coords)
        x1 = np.min(x_coords)
        y2 = np.max(y_coords)
        x2 = np.max(x_coords)


        roi = create_roi(
            rep,
            vectors=[
                FiveDVector(x=x1, y=y1, z=z, t=t, c=c),
                FiveDVector(x=x2, y=y1, z=z, t=t, c=c),
                FiveDVector(x=x2, y=y2, z=z, t=t, c=c),
                FiveDVector(x=x1, y=y2, z=z, t=t, c=c),
            ],
            type=RoiKind.RECTANGLE,
            label="size: {}".format(len(centrois)),
        )
        rois.append(roi)

    return rois




@register(collections=["organization"])
def merge_tables(table: List[Table]) -> Table:
    """Merge Tables

    Merges a list of tables into a single table

    Args:
        table (List[TableFragment]): The tables to merge

    Returns:
        TableFragment: The merged table
    """
    rep_origins = []
    for t in table:
        for o in t.rep_origins:
            rep_origins.append(o)

    frames = []

    for t in table:
        same = t.data
        same["origin"] = t.id
        frames.append(same)


    new_dataframe = pd.concat(frames, ignore_index=True)

    return from_parquet_like(new_dataframe, name="Merged Table")

