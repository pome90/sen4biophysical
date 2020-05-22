import geojson
import os 
import numpy as np

def biophysical(file_path, dst_dir, aoi_geojson):
    from sen4biophysical.utils import sentinel2_preprocessing, read_snappy_product, write_snappy_product

    from sen4biophysical.lai import LAI
    from sen4biophysical.cab import CAB
    from sen4biophysical.cw import CW
    from sen4biophysical.fapar import FAPAR
    from sen4biophysical.fcover import FCOVER
    
    if aoi_geojson:
        with open(aoi_geojson) as f:
            aoi_geojson = geojson.load(f)
    else:
        aoi_geojson = None

    data, metadata, mask, geo_coding = sentinel2_preprocessing(file_path=file_path, aoi_geojson=aoi_geojson)

    data = np.transpose(data, [1, 2, 0])
    
    viewZen_norm = np.cos(np.radians(metadata[0]))
    sunZen_norm = np.cos(np.radians(metadata[2]))
    relAzim_norm = np.cos(np.radians(metadata[3] - metadata[1]))

    metadata_arr = np.empty(shape=[data.shape[0], data.shape[1], 3], dtype="float32")
    metadata_arr[..., 0] = viewZen_norm
    metadata_arr[..., 1] = sunZen_norm
    metadata_arr[..., 2] = relAzim_norm

    target_shape = (data.shape[0], data.shape[1])

    data = np.concatenate([data, metadata_arr], axis=-1)
    data = np.reshape(data, [-1, 11])

    laiNet = LAI()
    cabNet = CAB()
    fcoverNet = FCOVER()
    cwNet = CW()
    faparNet = FAPAR()

    lai, lai_mask = laiNet(data)
    lai_mask = np.asarray(lai_mask, dtype=np.float32)
    lai = np.transpose(np.reshape(lai, (target_shape[0], target_shape[1], 1)), [2, 0, 1])
    lai_mask = np.transpose(np.reshape(lai_mask, (target_shape[0], target_shape[1], 1)), [2, 0, 1])
    
    cab, cab_mask = cabNet(data)
    cab_mask = np.asarray(cab_mask, dtype=np.float32)
    cab = np.transpose(np.reshape(cab, (target_shape[0], target_shape[1], 1)), [2, 0, 1])
    cab_mask = np.transpose(np.reshape(cab_mask, (target_shape[0], target_shape[1], 1)), [2, 0, 1])

    fcover, fcover_mask = fcoverNet(data)
    fcover_mask = np.asarray(fcover_mask, dtype=np.float32)
    fcover = np.transpose(np.reshape(fcover, (target_shape[0], target_shape[1], 1)), [2, 0, 1])
    fcover_mask = np.transpose(np.reshape(fcover_mask, (target_shape[0], target_shape[1], 1)), [2, 0, 1])

    cwater, cwater_mask = cwNet(data)
    cwater_mask = np.asarray(cwater_mask, dtype=np.float32)
    cwater = np.transpose(np.reshape(cwater, (target_shape[0], target_shape[1], 1)), [2, 0, 1])
    cwater_mask = np.transpose(np.reshape(cwater_mask, (target_shape[0], target_shape[1], 1)), [2, 0, 1])

    fapar, fapar_mask = faparNet(data)
    fapar_mask = np.asarray(fapar_mask, dtype=np.float32)
    fapar = np.transpose(np.reshape(fapar, (target_shape[0], target_shape[1], 1)), [2, 0, 1])
    fapar_mask = np.transpose(np.reshape(fapar_mask, (target_shape[0], target_shape[1], 1)), [2, 0, 1])

    bands = [
        dict(band_data=lai[0], band_name="lai"),
        dict(band_data=lai_mask[0], band_name="lai_mask"),
        dict(band_data=cab[0], band_name="cab", unit="g/cm2"),
        dict(band_data=cab_mask[0], band_name="cab_mask"),
        dict(band_data=fcover[0], band_name="fcover"),
        dict(band_data=fcover_mask[0], band_name="fcover_mask"),
        dict(band_data=cwater[0], band_name="cwater", unit="g/m2"),
        dict(band_data=cwater_mask[0], band_name="cwater_mask"),
        dict(band_data=fapar[0], band_name="fapar"),
        dict(band_data=fapar_mask[0], band_name="fapar_mask")
    ]

    product_name =  os.path.splitext(file_path.split("/")[-1])[0] + "_biophysical"
    save_path = os.path.join(dst_dir, product_name)

    write_snappy_product('{}'.format(save_path), bands, product_name, geo_coding=geo_coding)