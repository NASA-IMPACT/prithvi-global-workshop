import boto3
import json
import os
import gc
import geopandas as gpd
import rasterio
import time
import torch
import yaml

from fastapi import FastAPI, Request
from fastapi import FastAPI, status, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from lib.downloader import DOWNLOAD_FOLDER, Downloader
from lib.infer import Infer
from lib.post_process import PostProcess
from lib.trainer import Trainer
from lib.consts import BUCKET_NAME, LAYERS
from lib.utils import assumed_role_session

from rasterio.io import MemoryFile
from rasterio.merge import merge
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from shapely.geometry import shape
from skimage.morphology import disk, binary_closing

from starlette.middleware.cors import CORSMiddleware


app = FastAPI()

CONFIG_FILENAME = os.environ.get('S3_CONFIG_FILENAME')
CHECKPOINT_FILE = os.environ.get('CHECKPOINT_FILENAME')

def download_from_s3(s3_path, download_path):
    session = assumed_role_session()
    s3_connection = session.resource('s3')
    bucket = s3_connection.Bucket(BUCKET_NAME)
    filename = s3_path.split('/')[-1]
    file_path = f"{download_path}/{filename}"
    bucket.download_file(s3_path.replace(f's3://{BUCKET_NAME}', ''), file_path)
    return file_path


def load_model():
    config_file_path = download_from_s3(CONFIG_FILENAME)
    model_weights_path = download_from_s3(CHECKPOINT_FILE, 'models')
    infer = Infer(config_file_path, model_weights_path)
    with open(config_file_path) as config:
        config = yaml.safe_load(config)
    return {config['case']: infer}

MODELS = load_model()


def download_files(infer_date, layer, bounding_box):
    downloader = Downloader(infer_date, layer)
    return downloader.download_tiles(bounding_box)


def save_cog(mosaic, profile, transform, filename):
    profile.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "dtype": 'float32',
            "count": 1,
        }
    )
    with rasterio.open(filename, 'w', **profile) as raster:
        raster.write(mosaic[0], 1)
    output_profile = cog_profiles.get('deflate')
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile)

    # Dataset Open option (see gdalwarp `-oo` option)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="512",
    )
    with MemoryFile() as memory_file:
        cog_translate(
            filename,
            memory_file.name,
            output_profile,
            config=config,
            quiet=True,
            in_memory=True,
        )
        connection = boto3.client('s3')
        connection.upload_fileobj(memory_file, BUCKET_NAME, filename)

    return f"s3://{BUCKET_NAME}/{filename}"


def post_process(detections, transform):
    contours, shape = PostProcess.prepare_contours(detections)
    detections = PostProcess.extract_shapes(detections, contours, transform, shape)
    detections = PostProcess.remove_intersections(detections)
    return PostProcess.convert_to_geojson(detections)


def subset_geojson(geojson, bounding_box):
    geom = [shape(i['geometry']) for i in geojson]
    geom = gpd.GeoDataFrame({'geometry': geom})
    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [bounding_box[0], bounding_box[1]],
                [bounding_box[2], bounding_box[1]],
                [bounding_box[2], bounding_box[3]],
                [bounding_box[0], bounding_box[3]],
                [bounding_box[0], bounding_box[1]],
            ]
        ],
    }
    bbox = shape(bbox)
    bbox = gpd.GeoDataFrame({'geometry': [bbox]})
    return json.loads(geom.overlay(bbox, how='intersection').to_json())


def batch(tiles, spacing=40):
    length = len(tiles)
    for tile in range(0, length, spacing):
        yield tiles[tile : min(tile + spacing, length)]

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def infer_from_model(request: Request):
    instances = await request.json()

    model_id = instances['model_id']
    infer_date = instances['date']
    bounding_box = instances['bounding_box']
    final_geojson = infer(model_id, infer_date, bounding_box)
    return JSONResponse(content=jsonable_encoder(final_geojson))


def infer(model_id, infer_date, bounding_box):
    if model_id not in MODELS:
        response = {'statusCode': 422}
        return JSONResponse(content=jsonable_encoder(response))
    infer = MODELS[model_id]
    all_tiles = list()
    # geojson_list = list()
    # geojson = {'type': 'FeatureCollection', 'features': []}

    for layer in LAYERS:
        tiles = download_files(infer_date, layer, bounding_box)
        for tile in tiles:
            tile_name = tile
            if model_id == 'burn_scars':
                tile_name = tile_name.replace('.tif', '_scaled.tif')
            all_tiles.append(tile_name)

    start_time = time.time()
    mosaic = []
    s3_link = ''
    if all_tiles:
        try:
            torch.cuda.synchronize()
            results = list()
            profiles = list()
            with torch.no_grad():
                for tiles in batch(all_tiles):
                    batch_results, batch_profiles = infer.infer(tiles)
                    results.extend(batch_results)
                    profiles.extend(batch_profiles)
            memory_files = list()
            del infer
            torch.cuda.empty_cache()
            for index, profile in enumerate(profiles):
                memfile = MemoryFile()
                profile.update({
                    'count': infer.config['n_channels'],
                    'dtype': 'float32'
                })
                with memfile.open(**profile) as memoryfile:
                    memoryfile.write(results[index])
                memory_files.append(memfile.open())

            mosaic, transform = merge(memory_files)
            mosaic[0] = binary_closing(mosaic[0], disk(6))
            [memfile.close() for memfile in memory_files]
            prediction_filename = f"predictions/{start_time}-predictions.tif"

            s3_link = save_cog(mosaic, profile, transform, prediction_filename)

            # geojson = post_process(mosaic[0], transform)

            # for geometry in geojson:
            #     updated_geometry = PostProcess.convert_geojson(geometry)
            #     geojson_list.append(updated_geometry)
            # geojson = subset_geojson(geojson_list, bounding_box)
        except Exception as e:
            print('!!! infer error', infer_date, model_id, bounding_box, e)
            torch.cuda.empty_cache()
        print("!!! Infer Time:", time.time() - start_time)
    gc.collect()

    return {
        model_id: {'s3_link': s3_link}
    }

INFER = Infer(CONFIG_FILENAME, CHECKPOINT_FILE)

# MODEL = Trainer(CONFIG_FILENAME)

@app.post('/invocations')
async def invocations(request: Request):
    # model() is a hypothetical function that gets the inference output:
    params = request.json()
    model_resp = infer(params['model_id'], params['infer_date'], params['bounding_box'])
    # postprocess data and store as s3 file or send back a geojson

    response = Response(
        content=model_resp,
        status_code=status.HTTP_200_OK,
        media_type="text/plain",
    )
    return response

@app.post('/ping')
async def ping(request: Request):
    return { 'successCode': 200, 'message': 'pong'}
