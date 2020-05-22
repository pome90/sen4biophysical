from typing import Any, List, Dict

try:
   from snappy import ProductIO, Product
   from snappy import GPF
   from snappy import HashMap
   from snappy import jpy
   from snappy import ProductData
except ImportError as error:
    print("You must install manualy ESA snappy library")
    raise error
    
from geojson import GeoJSON
import geomet.wkt
import re
import os
import numpy as np


def geojson_to_wkt(geojson_obj: GeoJSON, feature_number: int = 0, decimals: int = 4) -> str:
    """Convert a GeoJSON object to Well-Known Text. Intended for use with OpenSearch queries.
    In case of FeatureCollection, only one of the features is used (the first by default).
    3D points are converted to 2D.
    Parameters
    ----------
    geojson_obj : dict
        a GeoJSON object
    feature_number : int, optional
        Feature to extract polygon from (in case of MultiPolygon
        FeatureCollection), defaults to first Feature
    decimals : int, optional
        Number of decimal figures after point to round coordinate to. Defaults to 4 (about 10
        meters).
    Returns
    -------
    polygon coordinates
        string of comma separated coordinate tuples (lon, lat) to be used by SentinelAPI
    """
    if "coordinates" in geojson_obj:
        geometry = geojson_obj
    elif "geometry" in geojson_obj:
        geometry = geojson_obj["geometry"]
    else:
        geometry = geojson_obj["features"][feature_number]["geometry"]

    def ensure_2d(geometry):
        if isinstance(geometry[0], (list, tuple)):
            return list(map(ensure_2d, geometry))
        else:
            return geometry[:2]

    def check_bounds(geometry):
        if isinstance(geometry[0], (list, tuple)):
            return list(map(check_bounds, geometry))
        else:
            if geometry[0] > 180 or geometry[0] < -180:
                raise ValueError("Longitude is out of bounds, check your JSON format or data")
            if geometry[1] > 90 or geometry[1] < -90:
                raise ValueError("Latitude is out of bounds, check your JSON format or data")

    # Discard z-coordinate, if it exists
    geometry["coordinates"] = ensure_2d(geometry["coordinates"])
    check_bounds(geometry["coordinates"])

    wkt = geomet.wkt.dumps(geometry, decimals=decimals)
    # Strip unnecessary spaces
    wkt = re.sub(r"(?<!\d) ", "", wkt)
    return wkt


def read_snappy_product(p: Product, bands: List[str] = None, dtype="float32") -> Any:
    width = p.getSceneRasterWidth()
    height = p.getSceneRasterHeight()

    if bands is None:
        bands = list(p.getBandNames())

    arr = np.empty((len(bands), height, width), dtype=dtype)
    for i, band_name in enumerate(bands):
        data = np.empty((height, width), dtype=dtype)
        band = p.getBand(band_name)

        try:
            band.readPixels(0, 0, width, height, data)
        except AttributeError:
            raise RuntimeError(p.getName() + " does not contain band " + band_name)

        arr[i] = data
    
    return arr


def write_snappy_product(file_path: str, bands: List[Dict], 
                         product_name: str, geo_coding: Any) -> None:
    try:
        (height, width) = bands[0]['band_data'].shape
    except AttributeError:
        raise RuntimeError(bands[0]['band_name'] + "contains no data.")
    product = Product(product_name, product_name, width, height)
    product.setSceneGeoCoding(geo_coding)

    # Ensure that output is saved in BEAM-DIMAP format, otherwise writeHeader does not work.
    file_path = os.path.splitext(file_path)[0] + '.dim'

    # Bands have to be created before header is written but header has to be written before band
    # data is written.
    for b in bands:
        band = product.addBand(b['band_name'], ProductData.TYPE_FLOAT32)
        if 'description' in b.keys():
            band.setDescription(b['description'])
        if 'unit' in b.keys():
            band.setUnit(b['unit'])
    product.setProductWriter(ProductIO.getProductWriter('BEAM-DIMAP'))
    product.writeHeader(str(file_path))
    for b in bands:
        band = product.getBand(b['band_name'])
        band.writePixels(0, 0, width, height, b['band_data'].astype(np.float32))
    product.closeIO()
    

def sentinel2_preprocessing(file_path: str, aoi_geojson: GeoJSON) -> List[Any]:
    p = ProductIO.readProduct(file_path)

    parameters = HashMap()
    parameters.put('targetResolution', '10')
    parameters.put('upsampling', 'Bilinear')
    
    p_resample = GPF.createProduct('Resample', parameters, p)

    parameters = HashMap()
    parameters.put('geoRegion', geojson_to_wkt(aoi_geojson))
    parameters.put('copyMetadata', 'true')
    parameters.put('referenceBand', 'B2')
    parameters.put('sourceBands', "B2,B3,B4,B5,B6,B7,B8A,B11,B12," + 
                                "quality_aot,quality_wvp,quality_cloud_confidence,quality_snow_confidence,quality_scene_classification," +
                                "view_zenith_mean,view_azimuth_mean,sun_zenith,sun_azimuth," + 
                                "view_zenith_B1,view_azimuth_B1,view_zenith_B2,view_azimuth_B2," + 
                                "view_zenith_B3,view_azimuth_B3,view_zenith_B4,view_azimuth_B4," + 
                                "view_zenith_B5,view_azimuth_B5,view_zenith_B6,view_azimuth_B6,view_zenith_B7," + 
                                "view_azimuth_B7,view_zenith_B8,view_azimuth_B8,view_zenith_B8A,view_azimuth_B8A," + 
                                "view_zenith_B9,view_azimuth_B9,view_zenith_B10,view_azimuth_B10,view_zenith_B11," + 
                                "view_azimuth_B11,view_zenith_B12,view_azimuth_B12")

    p_subset = GPF.createProduct("Subset", parameters, p_resample)

    width = p_subset.getSceneRasterWidth()
    height = p_subset.getSceneRasterHeight()

    parameters = HashMap()
    parameters.put('region', "0,0,{},{}".format(width, height))
    parameters.put('copyMetadata', 'true')
    parameters.put('referenceBand', 'B2')
    parameters.put('sourceBands', "B2,B3,B4,B5,B6,B7,B8A,B11,B12,view_zenith_mean,view_azimuth_mean,sun_zenith,sun_azimuth")

    p_reflectance = GPF.createProduct("Subset", parameters, p_subset)

    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')

    mask = BandDescriptor()
    mask.name = 'mask'
    mask.type = 'float32'
    mask.expression = 'if (quality_scene_classification >= 8 && quality_scene_classification <= 10) || quality_scene_classification == 3 then 0 else 1'
    mask.noDataValue = 0.0

    targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
    targetBands[0] = mask

    parameters = HashMap()
    parameters.put('targetBands', targetBands)

    p_mask = GPF.createProduct("BandMaths", parameters, p_subset)

    data = read_snappy_product(p_reflectance, bands=["B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"])
    metadata = read_snappy_product(p_reflectance, bands=["view_zenith_mean", "view_azimuth_mean", "sun_zenith", "sun_azimuth"])
    mask = read_snappy_product(p_mask)
    geo_coding = p_subset.getSceneGeoCoding()
    
    System = jpy.get_type('java.lang.System')
    System.gc()
    
    return data, metadata, mask, geo_coding