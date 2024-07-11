
from pathlib import Path

from torch_geometric.utils import from_networkx
from tqdm import tqdm

# from models.feat.bevcv.bevcv import BEVCV
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, OnDiskDataset
from torch.utils.data import Dataset
from pathlib import Path
import os
import re
import math
import io
import random
import time
import concurrent.futures
import threading
import hashlib
import requests
import shapefile
import shapely.geometry
from PIL import Image, ImageEnhance, ImageOps
import ray
from ray.experimental import tqdm_ray
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import torchvision.transforms as T
import torchvision
from streetview import search_panoramas
from streetview import get_panorama

torchvision.disable_beta_transforms_warning()
Image.MAX_IMAGE_PIXELS = None
TILE_SIZE = 256 
EARTH_CIRCUMFERENCE = 40075.016686 * 1000  
GOOGLE_MAPS_VERSION_FALLBACK = '934'
GOOGLE_MAPS_OBLIQUE_VERSION_FALLBACK = '148'
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15"
LOGGER = None
VERBOSITY = None

resize_pov = T.Resize((256, 256), antialias=True)
to_ten = T.ToTensor()

def download_sat(
        tile_path_template: str = str(Path.cwd() / 'data/images'),# + '/aerialbot/aerialbot-tiles/{angle_if_oblique}z{zoom}x{x}y{y}-{hash}.png',
        image_path_template: str = str(Path.cwd() / 'data/images'),# + '/{latitude},{longitude}-{width}x{height}m-z{zoom}-{image_height}x{image_width}.png',
        max_tries: int = 10,
        tile_url_template: str = "googlemaps",
        shapefile: str = "aerialbot/shapefiles/usa/usa.shp",
        point: tuple = (51.243594, -0.576837),
        width: int = 1000,
        height: int = 1000,
        image_width: int = 2048,
        image_height: int = 2048,
        max_meters_per_pixel: float = 0.4,
        apply_adjustments: bool = True,
        image_quality: int = 100,
        save: bool = True):

    tile_path_template += '/aerialbot/aerialbot-tiles/{angle_if_oblique}z{zoom}x{x}y{y}-{hash}.png'
    image_path_template += '/satellite/{latitude},{longitude}-{width}x{height}m-z{zoom}-{image_height}x{image_width}.png'

    direction = ViewDirection("downward")
    if tile_url_template == "googlemaps": tile_url_template = "https://khms2.google.com/kh/v={google_maps_version}?x={x}&y={y}&z={zoom}"
    elif tile_url_template == "navermap": tile_url_template = "https://map.pstatic.net/nrb/styles/satellite/{naver_map_version}/{zoom}/{x}/{y}.jpg?mt=bg"

    if "{google_maps_version}" in tile_url_template:
        google_maps_version = GOOGLE_MAPS_VERSION_FALLBACK
        if direction.is_oblique(): google_maps_version = GOOGLE_MAPS_OBLIQUE_VERSION_FALLBACK

        try:
            google_maps_page = requests.get("https://maps.googleapis.com/maps/api/js", headers={"User-Agent": USER_AGENT}).content
            match = re.search(rb'null,\[\[\"https:\/\/khms0\.googleapis\.com\/kh\?v=([0-9]+)', google_maps_page)
            if direction.is_oblique(): match = re.search(rb'\],\[\[\"https:\/\/khms0\.googleapis\.com\/kh\?v=([0-9]+)', google_maps_page)
            if match: google_maps_version = match.group(1).decode('ascii')
        except:
            print("Failed to determine current Google Maps version, falling back to", google_maps_version)
        tile_url_template = tile_url_template.replace("{google_maps_version}", google_maps_version)

    if "{naver_map_version}" in tile_url_template:
        naver_map_version = requests.get("https://map.pstatic.net/nrb/styles/satellite.json", headers={'User-Agent': USER_AGENT}).json()["version"]
        tile_url_template = tile_url_template.replace("{naver_map_version}", naver_map_version)

    MapTile.tile_path_template = tile_path_template
    MapTile.tile_url_template = tile_url_template

    geowidth = width
    geoheight = height
    foreshortening_factor = 1
    if direction.is_oblique(): foreshortening_factor = math.sqrt(2)

    # process max_meters_per_pixel setting
    if image_width is None and image_height is None: assert max_meters_per_pixel is not None
    elif image_height is None: max_meters_per_pixel = (max_meters_per_pixel or 1) * (width / image_width)
    elif image_width is None: max_meters_per_pixel = (max_meters_per_pixel or 1) * (height / image_height) / foreshortening_factor
    else:
        if width / image_width <= (height / image_height) / foreshortening_factor: max_meters_per_pixel = (max_meters_per_pixel or 1) * (width / image_width)
        else: max_meters_per_pixel = (max_meters_per_pixel or 1) * (height / image_height) / foreshortening_factor

    # process image width and height for scaling
    if image_width is not None or image_height is not None:
        if image_height is None: image_height = height * (image_width / width) / foreshortening_factor
        elif image_width is None: image_width = width * (image_height / height) * foreshortening_factor

    ############################################################################
    if shapefile is None and point is None: raise RuntimeError("neither shapefile path nor point configured")
    elif point is None: shapes = GeoShape(shapefile)

    for tries in range(0, max_tries):
        if tries > max_tries: raise RuntimeError("too many retries – maybe there's no internet connection? either that, or your max_meters_per_pixel setting is too low")
        if point is None: p = shapes.random_geopoint()
        else: p = GeoPoint(point[0], point[1])
        zoom = p.compute_zoom_level(max_meters_per_pixel)
        rect = GeoRect.around_geopoint(p, geowidth, geoheight)
        grid = MapTileGrid.from_georect(rect, zoom, direction)
        if point is not None: break

    ############################################################################
    grid.download()
    grid.stitch()
    image = MapTileImage(grid.image)
    image.crop(zoom, direction, rect)

    if image_width is not None or image_height is not None: image.scale(image_width, image_height)

    if apply_adjustments: image.enhance()

    image_path = image_path_template.format(latitude=p.lat, longitude=p.lon, width=width, height=height, zoom=zoom, image_height=image_height, image_width=image_width)

    d = os.path.dirname(image_path) 
    if not os.path.isdir(d): os.makedirs(d)
    image.save(image_path, image_quality)

    return image.image

def create_graph(centre=(51.509865, -0.118092), dist=1000, cwd='/home/ts00987/graphgl/data', fov=90, workers=4):
    g = ox.graph.graph_from_point(center_point=centre, dist=dist, dist_type='bbox', network_type='drive', simplify=True, retain_all=False, 
                                  truncate_by_edge=False, clean_periphery=None, custom_filter=None)
    g = ox.projection.project_graph(g, to_latlong=True)
    graph = nx.Graph()

    for n in g.nodes(data=True): 
        position = (n[1]['y'], n[1]['x'])
        graph.add_node(n[0], pos=position)

    for start, end in g.edges(): graph.add_edge(start, end)

    positions = nx.get_node_attributes(graph, 'pos')

    ray.init()
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    node_list = list(graph.nodes)
    bar = remote_tqdm.remote(total=len(node_list))
    node_lists = [node_list[i::workers] for i in range(workers)]
    street_getters = [download_junction_data.remote(node_lists[i], positions, fov, cwd, bar) for i in range(workers)]
    graph_images = ray.get(street_getters)
    bar.close.remote()
    # graph_images = download_junction_data(node_list=list(graph.nodes), positions=positions, fov=fov, cwd=cwd)
    graph_images = dict((key, d[key]) for d in graph_images for key in d)



    for node in graph_images.keys():
        graph.nodes[node]['sat'] = graph_images[node]['sat']
        graph.nodes[node]['pov'] = graph_images[node]['pov']
    return graph

@ray.remote
def download_junction_data(node_list, positions, fov, cwd, bar=None):
    missing = 0
    sub_dict = {}

    for node in tqdm(node_list, 'Downloading Junction Data') if bar is None else node_list:
        pos = positions[node]
        junction_sat = download_sat(tile_path_template='/home/ts00987/graphgl/data/images', image_path_template='/home/ts00987/graphgl/data/images', point=pos, width=100, height=100, image_width=256, image_height=256, save=False)
        crop = junction_sat.resize((256, 256))
        sat_crop = to_ten(crop).int()
        stret_crop = get_streetview(pos, fov=fov, cwd=cwd)
        if stret_crop is None: 
            stret_crop = torch.zeros((3, 256, 256)).int()
            missing += 1
        else: stret_crop = stret_crop.int().squeeze()
        sub_dict[node] = {'sat': sat_crop, 'pov': stret_crop}
        if bar is not None: bar.update.remote(1)

    return sub_dict

def random_walk(graph, start_node, length):
    walk = [start_node]
    while len(walk) < length:
        neighbours = list(graph.neighbors(walk[-1]))
        if len(walk) > 1:
            if walk[-2] in neighbours:
                neighbours.remove(walk[-2])
        if len(neighbours) == 0: break
        else: walk.append(random.choice(neighbours))
    walk = tuple(walk)
    return walk

def get_streetview(point, fov=90, cwd='/vol/research/deep_localisation/sat/'):
    try:
        pano = search_panoramas(lat=point[0], lon=point[1])
        dates = [d.date for d in pano if d.date is not None]
        latest = pano[dates.index(sorted([i for i in dates if i is not None])[-1])]
        pano = {'id': latest.pano_id, 'lat': latest.lat, 'lon': latest.lon, 'heading': latest.heading}
        image_name = f'{pano["id"]}.jpg'
        if Path(f'{cwd}/data/images/streetview/{image_name}').is_file(): 
            panorama = Image.open(f'{cwd}/data/images/streetview/{image_name}')
        else:
            panorama = get_panorama(pano_id=pano['id'])
            panorama.save(f'{cwd}/data/images/streetview/{image_name}')
        pano_width, pano_height = panorama.size
        half = math.floor(pano_width / 2)
        yaw = ((pano["heading"]+90)/360)
        crop_half_width = math.floor(((fov / 360) * pano_width) / 2)
        heading_pixel = math.floor(yaw * pano_width)
        height_bottom = math.floor(pano_height * 0.75)
        height_top = math.floor(pano_height * 0.25)

        if heading_pixel > crop_half_width and heading_pixel < (pano_width - crop_half_width):
            centre_point = heading_pixel
        else:
            left_img = (0, 0, half, pano_height)
            right_img = (half, 0, pano_width, pano_height)
            left_crop = panorama.crop(left_img)
            right_crop = panorama.crop(right_img)
            width1, height1 = left_crop.size
            width2, height2 = right_crop.size
            new_width = width1 + width2
            new_height = max(height1, height2)
            new_image = Image.new("RGB", (new_width, new_height))
            new_image.paste(right_crop, (0, 0))
            new_image.paste(left_crop, (width1, 0))
            if yaw < fov: centre_point = math.floor((((pano["heading"]+90+180)/360)) * pano_width)
            else: centre_point = math.floor((((pano["heading"]+90-180)/360)) * pano_width)

        left = math.floor(centre_point - crop_half_width)
        right = math.floor(centre_point + crop_half_width)
        area = (left, height_top, right, height_bottom)
        crop_img = panorama.crop(area)
        crop_ten = resize_pov(to_ten(crop_img))
    except:
        return None
    return crop_ten


class ViewDirection:
    def __init__(self, direction):
        self.angle = -1
        if direction == "downward": pass
        elif direction == "northward": self.angle = 0
        elif direction == "eastward": self.angle = 90
        elif direction == "southward": self.angle = 180
        elif direction == "westward": self.angle = 270
        else: raise ValueError(f"not a recognized view direction: {direction}")
        self.direction = direction

    def __repr__(self): return f"ViewDirection({self.direction})"

    def __str__(self): return self.direction

    def is_downward(self): return self.angle == -1

    def is_oblique(self): return not self.is_downward()

    def is_northward(self): return self.angle == 0

    def is_eastward(self): return self.angle == 90

    def is_southward(self): return self.angle == 180

    def is_westward(self): return self.angle == 270


class WebMercator:
    @staticmethod
    def project(geopoint, zoom):
        factor = (1 / (2 * math.pi)) * 2 ** zoom
        x = factor * (math.radians(geopoint.lon) + math.pi)
        y = factor * (math.pi - math.log(math.tan((math.pi / 4) + (math.radians(geopoint.lat) / 2))))
        return (x, y)


class ObliqueWebMercator:
    @staticmethod
    def project(geopoint, zoom, direction):
        x0, y0 = WebMercator.project(geopoint, zoom)
        width_and_height_of_world_in_tiles = 2 ** zoom
        equator_offset_from_edges = width_and_height_of_world_in_tiles / 2
        x, y = x0, y0
        if direction.is_northward(): pass
        elif direction.is_eastward():
            x = y0
            y = width_and_height_of_world_in_tiles - x0
        elif direction.is_southward():
            x = width_and_height_of_world_in_tiles - x0
            y = width_and_height_of_world_in_tiles - y0
        elif direction.is_westward():
            x = width_and_height_of_world_in_tiles - y0
            y = x0
        else: raise ValueError("direction must be one of 'northward', 'eastward', 'southward', or 'westward'")
        y = ((y - equator_offset_from_edges) / math.sqrt(2)) + equator_offset_from_edges
        return (x, y)


class GeoPoint:
    def __init__(self, lat, lon):
        assert -90 <= lat <= 90 and -180 <= lon <= 180
        self.lat = lat
        self.lon = lon

    def __repr__(self): return f"GeoPoint({self.lat}, {self.lon})"

    def fancy(self):
        def fancy_coord(coord, pos, neg):
            coord_dir = pos if coord > 0 else neg
            coord_tmp = abs(coord)
            coord_deg = math.floor(coord_tmp)
            coord_tmp = (coord_tmp - math.floor(coord_tmp)) * 60
            coord_min = math.floor(coord_tmp)
            coord_sec = round((coord_tmp - math.floor(coord_tmp)) * 600) / 10
            coord = f"{coord_deg}°{coord_min}'{coord_sec}\"{coord_dir}"
            return coord
        lat = fancy_coord(self.lat, "N", "S")
        lon = fancy_coord(self.lon, "E", "W")
        return f"{lat} {lon}"

    @classmethod
    def random(cls, georect):
        north = math.radians(georect.ne.lat)
        south = math.radians(georect.sw.lat)
        lat = math.degrees(math.asin(random.random() * (math.sin(north) - math.sin(south)) + math.sin(south)))
        west = georect.sw.lon
        east = georect.ne.lon
        width = east - west
        if width < 0: width += 360
        lon = west + width * random.random()
        if lon > 180: lon -= 360
        elif lon < -180: lon += 360
        return cls(lat, lon)

    def to_maptile(self, zoom, direction):
        x, y = WebMercator.project(self, zoom)
        if direction.is_oblique(): x, y = ObliqueWebMercator.project(self, zoom, direction)
        return MapTile(zoom, direction, math.floor(x), math.floor(y))

    def to_shapely_point(self): return shapely.geometry.Point(self.lon, self.lat)

    def compute_zoom_level(self, max_meters_per_pixel):
        meters_per_pixel_at_zoom_0 = ((EARTH_CIRCUMFERENCE / TILE_SIZE) * math.cos(math.radians(self.lat)))
        for zoom in reversed(range(0, 23+1)):
            meters_per_pixel = meters_per_pixel_at_zoom_0 / (2 ** zoom)
            if meters_per_pixel > max_meters_per_pixel: return zoom + 1
        else: raise RuntimeError("your settings seem to require a zoom level higher than is commonly available")


class GeoRect:
    def __init__(self, sw, ne):
        assert sw.lat <= ne.lat
        self.sw = sw
        self.ne = ne

    def __repr__(self): return f"GeoRect({self.sw}, {self.ne})"

    @classmethod
    def from_shapefile_bbox(cls, bbox):
        sw = GeoPoint(bbox[1], bbox[0])
        ne = GeoPoint(bbox[3], bbox[2])
        return cls(sw, ne)

    @classmethod
    def around_geopoint(cls, geopoint, width, height):
        assert width > 0 and height > 0
        meters_per_degree = (EARTH_CIRCUMFERENCE / 360)
        width_geo = width / (meters_per_degree * math.cos(math.radians(geopoint.lat)))
        height_geo = height / meters_per_degree
        southwest = GeoPoint(geopoint.lat - height_geo / 2, geopoint.lon - width_geo / 2)
        northeast = GeoPoint(geopoint.lat + height_geo / 2, geopoint.lon + width_geo / 2)
        return cls(southwest, northeast)

    def area(self):
        earth_radius = EARTH_CIRCUMFERENCE / (1000 * 2 * math.pi)
        earth_surface_area_in_km = 4 * math.pi * earth_radius ** 2
        spherical_cap_difference = (2 * math.pi * earth_radius ** 2) * abs(math.sin(math.radians(self.sw.lat)) - math.sin(math.radians(self.ne.lat)))
        area = spherical_cap_difference * (self.ne.lon - self.sw.lon) / 360
        assert area > 0 and area <= spherical_cap_difference and area <= earth_surface_area_in_km
        return area


class GeoShape:
    def __init__(self, shapefile_path):
        sf = shapefile.Reader(shapefile_path)
        self.shapes = sf.shapes()
        assert len(self.shapes) > 0
        assert all([shape.shapeTypeName == 'POLYGON' for shape in self.shapes])
        self.shapes_data = None

    def random_geopoint(self):
        if self.shapes_data is None:
            self.shapes_data = []
            for shape in self.shapes:
                bounds = GeoRect.from_shapefile_bbox(shape.bbox)
                area = GeoRect.area(bounds)
                self.shapes_data.append({"outline": shape, "bounds": bounds, "area": area, "area_relative_prefix_sum": 0})
            total = sum([shape["area"] for shape in self.shapes_data])
            area_prefix_sum = 0
            for shape in self.shapes_data:
                area_prefix_sum += shape["area"]
                shape["area_relative_prefix_sum"] = area_prefix_sum / total

        i = 0
        while i < 250:
            area_relative_prefix_sum = random.random()
            shape = None
            for shape_candidate in self.shapes_data:
                if area_relative_prefix_sum < shape_candidate["area_relative_prefix_sum"]:
                    shape = shape_candidate
                    break

            geopoint = GeoPoint.random(shape["bounds"])
            point = geopoint.to_shapely_point()
            polygon = shapely.geometry.shape(shape["outline"])
            contains = polygon.contains(point)
            if contains: return geopoint
        raise ValueError("cannot seem to find a point in the shape's bounding box that's within the shape – is your data definitely okay (it may well be if it's a bunch of spread-out islands)? if you're sure, you'll need to raise the iteration limit in this function")


class MapTileStatus:
    PENDING = 1
    CACHED = 2
    DOWNLOADING = 3
    DOWNLOADED = 4
    ERROR = 5


class MapTile:
    tile_path_template = None
    tile_url_template = None
    def __init__(self, zoom, direction, x, y):
        self.zoom = zoom
        self.direction = direction
        self.x = x
        self.y = y

        # initialize the other variables
        self.status = MapTileStatus.PENDING
        self.image = None
        self.filename = None
        if (MapTile.tile_path_template):
            self.filename = MapTile.tile_path_template.format(
                angle_if_oblique=("" if self.direction.is_downward() else f"deg{self.direction.angle}"),
                zoom=self.zoom,
                x=self.x,
                y=self.y,
                hash=hashlib.sha256(MapTile.tile_url_template.encode("utf-8")).hexdigest()[:8])

    def __repr__(self): return f"MapTile({self.zoom}, {self.direction}, {self.x}, {self.y})"

    def zoomed(self, zoom_delta):
        zoom = self.zoom + zoom_delta
        fac = (2 ** zoom_delta)
        return MapTileGrid([[MapTile(zoom, self.direction, self.x * fac + x, self.y * fac + y) for y in range(0, fac)] for x in range(0, fac)])

    def load(self):
        if self.filename is None: self.download()
        else:
            try:
                self.image = Image.open(self.filename)
                self.image.load()
                self.status = MapTileStatus.CACHED
            except IOError:
                self.download()

    def download(self):
        self.status = MapTileStatus.DOWNLOADING
        try:
            url = MapTile.tile_url_template.format(angle=self.direction.angle, x=self.x, y=self.y, zoom=self.zoom)
            r = requests.get(url, headers={'User-Agent': USER_AGENT})
        except requests.exceptions.ConnectionError:
            self.status = MapTileStatus.ERROR
            return

        if r.status_code != 200:
            LOGGER.warning(f"Unable to download {self}, status code {r.status_code}.")
            self.status = MapTileStatus.ERROR
            return
        
        data = r.content
        self.image = Image.open(io.BytesIO(data))

        assert self.image.mode == "RGB"
        assert self.image.size == (TILE_SIZE, TILE_SIZE)

        if self.filename is not None:
            d = os.path.dirname(self.filename)
            if not os.path.isdir(d):
                os.makedirs(d)
            with open(self.filename, 'wb') as f:
                f.write(data)
        self.status = MapTileStatus.DOWNLOADED


class ProgressIndicator:
    def __init__(self, maptilegrid):
        self.maptilegrid = maptilegrid

    def update_tile(self, maptile):
        def p(s): print(s + "\033[0m", end='')
        if maptile.status == MapTileStatus.PENDING: p("░░")
        elif maptile.status == MapTileStatus.CACHED: p("\033[34m" + "██")
        elif maptile.status == MapTileStatus.DOWNLOADING: p("\033[33m" + "▒▒")
        elif maptile.status == MapTileStatus.DOWNLOADED: p("\033[32m" + "██")
        elif maptile.status == MapTileStatus.ERROR: p("\033[41m\033[37m" + "XX")

    def update_text(self):
        cached = 0
        downloaded = 0
        errors = 0
        for maptile in self.maptilegrid.flat():
            if maptile.status == MapTileStatus.CACHED: cached += 1
            elif maptile.status == MapTileStatus.DOWNLOADED: downloaded += 1
            elif maptile.status == MapTileStatus.ERROR: errors += 1

        done = cached + downloaded
        total = self.maptilegrid.width * self.maptilegrid.height
        percent = int(10 * (100 * done / total)) / 10

        details = f"{done}/{total}"
        if cached: details += f", {cached} cached"
        if downloaded: details += f", {downloaded} downloaded"
        if errors:
            details += f", {errors} error"
            if errors > 1: details += "s"
        print(f"{percent}% ({details})")

    def update(self):
        if VERBOSITY == "normal":
            self.update_text()
            return

        for y in range(self.maptilegrid.height):
            for x in range(self.maptilegrid.width):
                maptile = self.maptilegrid.at(x, y)
                self.update_tile(maptile)
            print()  # line break

        self.update_text()
        print(f"\033[{self.maptilegrid.height + 1}A", end='')

    def loop(self):
        if VERBOSITY == "quiet": return

        while any([maptile.status is MapTileStatus.PENDING or
                   maptile.status is MapTileStatus.DOWNLOADING
                   for maptile in self.maptilegrid.flat()]):
            self.update()
            time.sleep(0.1)
        self.update()  # final update to show that we're all done

    def cleanup(self):
        if VERBOSITY == "quiet" or VERBOSITY == "normal": return
        print(f"\033[{self.maptilegrid.height}B")


class MapTileGrid:
    def __init__(self, maptiles):
        self.maptiles = maptiles
        self.width = len(maptiles)
        self.height = len(maptiles[0])
        self.image = None

    def __repr__(self):
        return f"MapTileGrid({self.maptiles})"

    @classmethod
    def from_georect(cls, georect, zoom, direction):
        bottomleft = georect.sw.to_maptile(zoom, direction)
        topright = georect.ne.to_maptile(zoom, direction)
        if bottomleft.x > topright.x: bottomleft.x, topright.x = topright.x, bottomleft.x
        if bottomleft.y < topright.y: bottomleft.y, topright.y = topright.y, bottomleft.y

        maptiles = []
        for x in range(bottomleft.x, topright.x + 1):
            col = []
            for y in range(topright.y, bottomleft.y + 1):
                maptile = MapTile(zoom, direction, x, y)
                col.append(maptile)
            maptiles.append(col)

        return cls(maptiles)

    def at(self, x, y):
        if x < 0: x += self.width
        if y < 0: y += self.height
        return self.maptiles[x][y]

    def flat(self): return [maptile for col in self.maptiles for maptile in col]

    def has_high_quality_imagery(self, quality_check_delta):
        corners = [self.at(x, y).zoomed(quality_check_delta).at(x, y) for x in [0, -1] for y in [0, -1]]
        all_good = True
        for c in corners:
            c.load()
            if c.status == MapTileStatus.ERROR:
                all_good = False
                break
        return all_good

    def download(self):
        # prog = ProgressIndicator(self)
        prog_thread = threading.Thread()#target=prog.loop)
        prog_thread.start()
        tiles = self.flat()
        random.shuffle(tiles)
        threads = max(self.width, self.height)
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor: {executor.submit(maptile.load): maptile for maptile in tiles}

        missing_tiles = [maptile for maptile in self.flat() if maptile.status == MapTileStatus.ERROR]
        if 0 < len(missing_tiles) < 0.02 * len(self.flat()):
            if VERBOSITY != "quiet": print("Retrying missing tiles...")
            for maptile in missing_tiles: maptile.load()

        prog_thread.join()
        # prog.cleanup()

        missing_tiles = [maptile for maptile in self.flat() if maptile.status == MapTileStatus.ERROR]
        if missing_tiles: raise RuntimeError(f"unable to load one or more map tiles: {missing_tiles}")

    def stitch(self):
        image = Image.new('RGB', (self.width * TILE_SIZE, self.height * TILE_SIZE))
        for x in range(0, self.width):
            for y in range(0, self.height): image.paste(self.maptiles[x][y].image, (x * TILE_SIZE, y * TILE_SIZE))
        self.image = image


class MapTileImage:
    def __init__(self, image):
        self.image = image

    def save(self, path, quality=90):
        self.image.save(path, quality=quality)

    def crop(self, zoom, direction, georect):
        left, bottom = WebMercator.project(georect.sw, zoom)  # sw_x, sw_y
        right, top = WebMercator.project(georect.ne, zoom)  # ne_x, ne_y
        if direction.is_oblique():
            left, bottom = ObliqueWebMercator.project(georect.sw, zoom, direction)
            right, top = ObliqueWebMercator.project(georect.ne, zoom, direction)

        if left > right: left, right = right, left
        if bottom < top: bottom, top = top, bottom

        left_crop = round(TILE_SIZE * (left % 1))
        bottom_crop = round(TILE_SIZE * (1 - bottom % 1))
        right_crop = round(TILE_SIZE * (1 - right % 1))
        top_crop = round(TILE_SIZE * (top % 1))

        crop = (left_crop, top_crop, right_crop, bottom_crop)
        self.image = ImageOps.crop(self.image, crop)

    def scale(self, width, height): self.image = self.image.resize((round(width), round(height)), resample=Image.Resampling.LANCZOS)

    def enhance(self):
        contrast = 1.07
        brightness = 1.01
        self.image = ImageEnhance.Contrast(self.image).enhance(contrast)
        self.image = ImageEnhance.Brightness(self.image).enhance(brightness)


class GraphData():
    def __init__(self, args):
        self.args = args
        self.download()
        self.prepare_graph()
        
    def prepare_graph(self):
        # Split graph into train and test - could use one quadrant as fully disconnected test set
        poses = []
        for node in self.corpus_graph.nodes: poses.append(self.corpus_graph.nodes[node]["pos"])
        poses = list(set(poses))
        lats, lons = [p[0] for p in poses], [p[1] for p in poses]

        # Get midpoint of graph - selecting lower right quadrant as test set
        min_lat, min_lon, max_lat, max_lon = min(lats), min(lons), max(lats), max(lons)
        mid_lat, mid_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
        mid_pos = (mid_lat, mid_lon)

        # Seperate nodes into train and test
        train_nodes, test_nodes = [], []
        for node in self.corpus_graph.nodes:
            pos = self.corpus_graph.nodes[node]["pos"]
            if pos[0] > mid_pos[0] and pos[1] > mid_pos[1]: test_nodes.append(node)
            else: train_nodes.append(node)

        self.train_graph = nx.Graph()
        for node in train_nodes: 
            self.train_graph.add_node(node)
            self.train_graph.nodes[node]['sat'] = self.corpus_graph.nodes[node]['sat']
            self.train_graph.nodes[node]['pov'] = self.corpus_graph.nodes[node]['pov']
            self.train_graph.nodes[node]['pos'] = self.corpus_graph.nodes[node]['pos']

        for node in train_nodes:
            for neighbour in self.corpus_graph.neighbors(node):
                if neighbour in train_nodes: self.train_graph.add_edge(node, neighbour)

        self.test_graph = nx.Graph()
        for node in test_nodes: 
            self.test_graph.add_node(node)
            self.test_graph.nodes[node]['sat'] = self.corpus_graph.nodes[node]['sat']
            self.test_graph.nodes[node]['pov'] = self.corpus_graph.nodes[node]['pov']
            self.test_graph.nodes[node]['pos'] = self.corpus_graph.nodes[node]['pos']

        for node in test_nodes:
            for neighbour in self.corpus_graph.neighbors(node):
                if neighbour in test_nodes: self.test_graph.add_edge(node, neighbour)

        train_walk_name = f'{self.args.path}/data/graphs/walks/train_walks_{self.args.point}_{self.args.width}_{self.args.fov}_{self.args.walk}.npy'
        test_walk_name = f'{self.args.path}/data/graphs/walks/test_walks_{self.args.point}_{self.args.width}_{self.args.fov}_{self.args.walk}.npy'

        if Path(train_walk_name).is_file() and Path(test_walk_name).is_file():
            self.train_walks = np.load(train_walk_name, allow_pickle=True)
            self.test_walks = np.load(test_walk_name, allow_pickle=True)
        else:
            self.train_walks = self.exhaustive_walks(self.train_graph, train_nodes, 'Train')
            self.test_walks = self.exhaustive_walks(self.test_graph, test_nodes, 'Test')
            np.save(train_walk_name, self.train_walks)
            np.save(test_walk_name, self.test_walks)



    def download(self):
        # check file exists
        if not Path(f'{self.args.path}/data/graphs/raw/graph_{self.args.point}_{self.args.width}_{self.args.fov}.pt').is_file():
            self.corpus_graph = create_graph(centre=self.args.point, dist=self.args.width, cwd=self.args.path, fov=self.args.fov, workers=self.args.workers)
            torch.save(self.corpus_graph, f'{self.args.path}/data/graphs/raw/graph_{self.args.point}_{self.args.width}_{self.args.fov}.pt')
        else: self.corpus_graph = torch.load(f'{self.args.path}/data/graphs/raw/graph_{self.args.point}_{self.args.width}_{self.args.fov}.pt')

    def exhaustive_walks(self, graph, nodes, stage='Train'):
        attempts = 100000
        node_walks = []
        for node in tqdm(nodes, desc=f'Generating {stage} Walks'):
            for attempts in range(attempts):
                walk = random_walk(graph, node, self.args.walk) # tuple not list
                if walk not in node_walks and len(walk) == self.args.walk:
                    node_walks.append(walk)            
        return node_walks

class GraphDataset(Dataset):
    def __init__(self, args, graph, stage='train'):
        self.args = args
        self.graph = graph.train_graph if stage == 'train' else graph.test_graph
        self.walks = graph.train_walks if stage == 'train' else graph.test_walks
        self.stage = stage

    def __len__(self):
        return len(self.walks)

    def __getitem__(self, idx):
        walk_nodes = self.walks[idx]
        walk = self.graph.subgraph(walk_nodes)
        # add start point        
        nx.set_node_attributes(walk, walk.nodes[walk_nodes[0]]['pos'], 'start_point')
        walk = from_networkx(walk)
        return walk


    
    
