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

Image.MAX_IMAGE_PIXELS = None
TILE_SIZE = 256 
EARTH_CIRCUMFERENCE = 40075.016686 * 1000  
GOOGLE_MAPS_VERSION_FALLBACK = '934'
GOOGLE_MAPS_OBLIQUE_VERSION_FALLBACK = '148'
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15"
LOGGER = None
VERBOSITY = None

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
        prog = ProgressIndicator(self)
        prog_thread = threading.Thread(target=prog.loop)
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
        prog.cleanup()

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


################################################ FUNCTION

def download_sat(
        tile_path_template: str = str(Path.cwd() / 'data/images') + '/aerialbot/aerialbot-tiles/{angle_if_oblique}z{zoom}x{x}y{y}-{hash}.png',
        image_path_template: str = str(Path.cwd() / 'data/images') + '/{latitude},{longitude}-{width}x{height}m-z{zoom}-{image_height}x{image_width}.png',
        max_tries: int = 10,
        tile_url_template: str = "googlemaps",
        shapefile: str = "aerialbot/shapefiles/usa/usa.shp",
        point: tuple = (51.243594, -0.576837),
        width: int = 1000,
        height: int = 1000,
        image_width: int = 2048,
        image_height: int = 2048,
        max_meters_per_pixel: float = 0.2,
        apply_adjustments: bool = True,
        image_quality: int = 100,
        save: bool = True):

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
    if direction.is_eastward() or direction.is_westward(): geowidth, geoheight = geoheight, geowidth

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

    image_path = image_path_template.format(
        latitude=p.lat,
        longitude=p.lon,
        width=width,
        height=height,
        zoom=zoom,
        image_height=image_height,
        image_width=image_width)

    if save:
        d = os.path.dirname(image_path) 
        if not os.path.isdir(d): os.makedirs(d)
        image.save(image_path, image_quality)

    return image.image






# Central London
# point = (51.511932, -0.112624)
# width, height = 6000, 6000
# image_height, image_width = None, None
# file_name = download_sat(point=point, width=width, height=height, image_width=image_width, image_height=image_height, max_meters_per_pixel=0.1)
