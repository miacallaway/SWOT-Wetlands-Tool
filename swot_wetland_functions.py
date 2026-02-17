'''
Functions to extract SWOT water surface elevation (WSE) from the SWOT Pixel Cloud product and plot water level time series relative to a gauge (Australian Bureau of Meteorology (BOM) gauge text files), 
spatially plot gridded water surface elevation or water surface elevation anomalies. Functions are also provided to compute spatial water depth where LiDAR data or a DEM is available. 

The functions can be used more generally for other wetland sites globally by substituting the parameters defined in the 'site_specific_variables.py' file. Examples for the wetlands we used are shown. 

The functions were created during the production of the paper 'SWOT reveals flood depths and environmental flows in wetlands' - Callaway et al., submitted October 2025

Authors: Mia Callaway, Maya Taib and Lachlan Dodd
Contact emails: Mia.Callaway@anu.edu.au, Maya.Taib@anu.edu.au, Paul.Tregoning@anu.edu.au
Research School of Earth Sciences
The Australian National University
'''

#########################
## Import Dependencies ##
#########################
from site_specific_variables import *
import numpy as np
import pandas as pd
import netCDF4 as nc
from pyproj import Transformer
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm
from shapely.ops import transform
from shapely import Polygon, STRtree, vectorized, box, Polygon, Point, LineString
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from tqdm import tqdm
import pickle
import math
import shapely
import os
import laspy

############################################
## General Processing Functions/Variables ##
############################################

img = mpimg.imread(image_file)
transformer = Transformer.from_crs("EPSG:4326", epsg_xy_m, always_xy=True)           # Transformer from WGS84 to local x,y (m units) projection system
transformer_reverse = Transformer.from_crs(epsg_xy_m, "EPSG:4326", always_xy=True)   # Transformer from WGS84 to local x,y (m units) projection system
wgs_to_gda_ahd = Transformer.from_crs('EPSG:4979', epsg_3D_vertical, always_xy=True) # EPSG:4979 WGS 84 (3D) to relevant 3D datum matching LiDAR/DEM height (e.g. EPSG:9464 is GDA94 + AHD71 height)
blue_white_green_cmap = LinearSegmentedColormap.from_list("blue_white_green", ["#2166ac", 'white', "#21ac2a"])

def colour_sequence(n, cmap_name):
    '''
    Creates a colour sequence of a given number of colours from an Matplotlib colourmap.
    
    Inputs:
        n: length of colour sequence (integer)
        cmap_name: name of colourmap in '' (e.g. 'coolwarm')
    
    Returns: list of n colours derived from that colourmap
    '''
    cmap = plt.cm.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]

def combinations(num_list):
    '''
    Obtains all possible summative combinations of a given list of numbers.
    
    Inputs:
        num_list: list of numbers to obtain summative combinations of
    
    Returns: list of numbers of all possible summative combinations of the provided list of numbers
    '''
    combinations = {0}
    for num in num_list:
        combinations |= {num + c for c in combinations}
    return combinations

geolocation_qual_combinations = combinations([0, 1, 64, 524288])
classification_qual_combinations_including_noprior = combinations([0, 1, 4, 16, 2048, 524288])
classification_qual_combinations_excluding_noprior = combinations([0, 1, 16, 2048, 524288])

def read_pixc_values_tide(ds1):
    '''
    Reads in all relevant variables from the SWOT file into numpy arrays.
    
    Inputs: 
        ds1: SWOT file opened using netCDF4

    Returns: list of arrays for variables required from the SWOT file
    '''
    height = (ds1.groups["pixel_cloud"].variables["height"][:])
    latitude = (ds1.groups["pixel_cloud"].variables["latitude"][:])
    longitude = (ds1.groups["pixel_cloud"].variables["longitude"][:])
    classification = (ds1.groups["pixel_cloud"].variables["classification"][:])
    sigma0 = (ds1.groups["pixel_cloud"].variables["sig0"][:])
    geoid = (ds1.groups["pixel_cloud"].variables["geoid"][:])
    intf_qual = (ds1.groups["pixel_cloud"].variables["interferogram_qual"][:])
    geolocation_qual = (ds1.groups["pixel_cloud"].variables["geolocation_qual"][:])
    pixel_area = (ds1.groups["pixel_cloud"].variables["pixel_area"][:])
    classification_qual = (ds1.groups["pixel_cloud"].variables["classification_qual"][:])
    sig0_qual = (ds1.groups["pixel_cloud"].variables["sig0_qual"][:])
    solid_earth_tide = (ds1.groups["pixel_cloud"].variables["solid_earth_tide"][:])
    load_tide_fes = (ds1.groups["pixel_cloud"].variables["load_tide_fes"][:])
    load_tide_got = (ds1.groups["pixel_cloud"].variables["load_tide_got"][:])
    pole_tide = (ds1.groups["pixel_cloud"].variables["pole_tide"][:])
    return height, latitude, longitude, classification, sigma0, geoid, intf_qual, geolocation_qual, pixel_area, classification_qual, sig0_qual, solid_earth_tide, load_tide_fes, load_tide_got, pole_tide

def mask_with_no_prior_included(classification, sigma0, intf_qual, geolocation_qual, classification_qual, sig0_qual):
    '''
    Masks out non-water areas and bad quality data from the arrays. 
    Inputs: series of numpy arrays used to mask the data
    Returns: array of 1 and NaN, where NaN values represent bad quality data
    '''
    mask = np.zeros(len(classification))
    for i in range(len(classification)):
        if (
            classification[i] in (3,4,5)
            and intf_qual[i] in (0, 524288)
            and geolocation_qual[i] in geolocation_qual_combinations
            and classification_qual[i] in classification_qual_combinations_including_noprior
            and sig0_qual[i] in (0, 524288)
        ):
            mask[i] = 1
        else:
            mask[i] = np.nan
    return mask

def mask_with_no_prior_excluded(classification, sigma0, intf_qual, geolocation_qual, classification_qual, sig0_qual):
    '''
    Masks out non-water areas and bad quality data from the arrays. Data flagged as 'detected_water_but_no_prior_water' is assigned a NaN. 
    Inputs: series of numpy arrays used to mask the data
    Returns: array of 1 and NaN, where NaN values represent bad quality data
    '''
    mask = np.zeros(len(classification))
    for i in range(len(classification)):
        if (
            classification[i] in (3,4,5)
            and intf_qual[i] in (0, 524288)
            and geolocation_qual[i] in geolocation_qual_combinations
            and classification_qual[i] in classification_qual_combinations_excluding_noprior
            and sig0_qual[i] in (0, 524288)
        ):
            mask[i] = 1
        else:
            mask[i] = np.nan
    return mask

def Maubant_C(classification, sigma0, intf_qual, geolocation_qual, classification_qual, sig0_qual):
    '''
    Masks out non-water areas and bad quality data from the arrays (as used in the Maubant et al. (2025) paper "Assessing the Accuracy of SWOT measurements of water bodies in Australia"). 
    Inputs: series of numpy arrays used to mask the data
    Returns: array of 1 and NaN, where NaN values represent bad quality data
    '''
    mask = np.zeros(len(classification))
    for i in range(len(classification)):
        if (
            2 < classification[i] < 5 
            and 36 < 10*np.log10(sigma0[i]**2) 
            and intf_qual[i] < 1
            and geolocation_qual[i] < 1
            and classification_qual[i] < 1
            and sig0_qual[i] < 1
        ):
            mask[i] = 1
        else:
            mask[i] = np.nan
    return mask

def Maubant_D(classification, sigma0, intf_qual, geolocation_qual, classification_qual, sig0_qual):
    '''
    Function to mask out non-water areas and bad quality data from the arrays (as used in the Maubant et al. (2025) paper titled "Assessing the Accuracy of SWOT measurements of water bodies in Australia").
    The sigma0 threshold value has been adjusted to account for a corresponding adjustment of a 2.5 dB absolute bias in SWOT version D data.
    Inputs: series of numpy arrays used to mask the data
    Returns: array of 1 and NaN, where NaN values represent bad quality data
    '''
    mask = np.zeros(len(classification))
    for i in range(len(classification)):
        if (
            2 < classification[i] < 5 
            and 33.5 < 10*np.log10(sigma0[i]**2) 
            and intf_qual[i] < 1
            and geolocation_qual[i] < 1
            and classification_qual[i] < 1
            and sig0_qual[i] < 1
        ):
            mask[i] = 1
        else:
            mask[i] = np.nan
    return mask

def get_wetland_bounding_box(wetland_polygons):
    '''Gets wetland polygon bounds (bounds of rectangle containing wetland polygon) for a series of wetland polygons. 
    Inputs:
        wetland_polygons: dictionary of wetland polygons

    Returns: 
        wetland_bounds: list of wetland bounds in the format [minimum x value, maximum x value, minimum y value, maximum y value]
    '''
    wetland_bounds = []
    for wetland_poly in wetland_polygons:
        x,y = wetland_polygons[wetland_poly].exterior.xy
        bounds = [np.min(x), np.max(x), np.min(y), np.max(y)]
        wetland_bounds.append(bounds)
    return wetland_bounds

def gridded_sum(grid_bounds, cell_size, x, y, values):
    """
    Grids data in 2D from arrays of x, y and variable of interest, with the sum of values assigned per grid cell. 

    Inputs:
        grid_bounds: minimum x and y bounds in the format [min_x, min_y, max_x, max_y].
        cell_size: Grid cell resolution (in m)
        x: x coordinates (in m)
        y: y coordinates (in m)
        values: Variable to sum.

    Returns:
        sum_grid (2D array): Grid of summed values.
    """
    min_x, min_y, max_x, max_y = grid_bounds
    width = int(np.ceil((max_x - min_x) / cell_size))
    height = int(np.ceil((max_y - min_y) / cell_size))

    in_bounds = ((x >= min_x) & (x < max_x) & (y >= min_y) & (y < max_y))
    x = x[in_bounds]
    y = y[in_bounds]
    values = values[in_bounds]

    col_indices = ((x - min_x) / cell_size).astype(int)
    row_indices = ((max_y - y) / cell_size).astype(int)
    col_indices = np.clip(col_indices, 0, width - 1)
    row_indices = np.clip(row_indices, 0, height - 1)

    sum_grid = np.zeros((height, width), dtype=np.float64)
    flat_idx = row_indices * width + col_indices
    np.add.at(sum_grid.ravel(), flat_idx, values)
    return sum_grid

########################################
## Gridded water depth over full site ##
########################################

def lidar_avg_grid(las_dir, grid_bounds, cell_size, output_npy_path, value_field='z'):
    '''
    Creates a grid of average elevation from a LiDAR point cloud
    
    Inputs:
        las_dir: directory containing LiDAR point cloud files (.laz format) 
        grid_bounds: bounds of area to grid (in m units) in the format [minimum x value, minimum y value, maximum x value, maximum y value]
        cell_size:
        output_npy_path
        value_field: value to average (set by default to the vertical axis/height variable 'z')

    Returns: averaged elevation grid
    '''
    min_x, min_y, max_x, max_y = grid_bounds
    width = int(np.ceil((max_x - min_x) / cell_size))
    height = int(np.ceil((max_y - min_y) / cell_size))
    
    sum_grid = np.zeros((height, width), dtype=np.float64)
    count_grid = np.zeros((height, width), dtype=np.uint32)

    for file in tqdm(os.listdir(las_dir), desc="Processing LAS files"):
        if not file.lower().endswith(('.las', '.laz')):
            continue
        filepath = os.path.join(las_dir, file)

        try:
            with laspy.open(filepath) as fh:
                header = fh.header
                las_bounds = (header.mins[0], header.mins[1], header.maxs[0], header.maxs[1])
                file_box = box(*las_bounds)
                grid_box = box(min_x, min_y, max_x, max_y)

                if not file_box.intersects(grid_box):
                    continue

                las = fh.read()
                x = np.array(las.x)
                y = np.array(las.y)
                values = np.array(getattr(las, value_field))

                in_bounds = ((x >= min_x) & (x < max_x) & (y >= min_y) & (y < max_y))
                x = x[in_bounds]
                y = y[in_bounds]
                values = values[in_bounds]

                if len(values) == 0:
                    continue

                col_indices = ((x - min_x) / cell_size).astype(int)
                row_indices = ((max_y - y) / cell_size).astype(int)
                col_indices = np.clip(col_indices, 0, width - 1)
                row_indices = np.clip(row_indices, 0, height - 1)

                flat_idx = row_indices * width + col_indices
                np.add.at(sum_grid.ravel(), flat_idx, values)
                np.add.at(count_grid.ravel(), flat_idx, 1)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_grid = sum_grid / count_grid
        avg_grid[count_grid == 0] = np.nan

    np.save(output_npy_path, avg_grid)
    print(f"Saved average grid to: {output_npy_path}")
    return avg_grid

def process_transformed_swot_data(file, wetland_mask):
    '''
    Takes in a SWOT filename and undertake basic processing to output good quality data within the broad study site (as defined in 'Site_specific_variables.py'). 
    
    Inputs:
        file: SWOT data filename
        wetland_mask: function to mask out bad data

    Returns: a series of arrays of different SWOT variables after poor quality data is masked out (variables: longitude, latitude, easting, northing, water_height (in same vertical datum as LiDAR/DEM data), area and area_weighted (area * water height))
    '''
    ds1 = nc.Dataset(f'{file}')
    data = read_pixc_values_tide(ds1)
    study_site_mask = (data[1] > ymin) & (data[1] < ymax) & (data[2] > xmin) & (data[2] < xmax)
    filtered_arrays = [array[study_site_mask] for array in data]
    [height, latitude, longitude, classification, sigma0, geoid, intf_qual, geolocation_qual, pixel_area, classification_qual, sig0_qual, solid_earth_tide, load_tide_fes, load_tide_got, pole_tide] = filtered_arrays

    mask = wetland_mask(classification, sigma0, intf_qual, geolocation_qual, classification_qual, sig0_qual) # Create mask to remove bad data
    Masked_Pixels = (height - solid_earth_tide - load_tide_fes - pole_tide) * mask        # Apply mask and tidal models to get ellipsoidal height: Note that load_tide_fes is used, not load_tide_got

    h = Masked_Pixels[~np.isnan(Masked_Pixels)]
    a = pixel_area[~np.isnan(Masked_Pixels)]
    lat = latitude[~np.isnan(Masked_Pixels)]
    lon = longitude[~np.isnan(Masked_Pixels)]

    if len(lon) > 0:
        east, north = transformer.transform(lon, lat)
        gda_lon, gda_lat, swot_ahd = wgs_to_gda_ahd.transform(lon, lat, h)
        area_weighted = swot_ahd * a
        return lon, lat, east, north, swot_ahd, a, area_weighted

    else:
        return [], [], [], [], [], [], []
    
def full_site_water_depth(directory, file_list, elevation_grid, min_x, min_y, max_x, max_y, wetland_cross_sections, size=100, wetland_mask=mask_with_no_prior_included):
    '''
    Computes water depth across a full study site. KDE filtering is applied over the full site after masking and the gridded elevation data is subtracted from 
    the gridded SWOT water surface elevation data to obtain water depth, with negative depth values removed. Wetland cross-sections are shown over the spatial plot for the first file. 
    The spatial plots are saved to a .png file which is labelled based on the SWOT file date. If gridded water surface elevation or gridded elevation is of interest, the sections 
    that are currently commented out can be uncommented to also plot gridded water surface elevation and gridded elevation and save as .png files. The latitude, longitude, elevation, 
    water surface elevation and depth data used to create the plots are also saved to numpy arrays. 
    
    Inputs:
        directory: directory containing SWOT files
        file_list: list of SWOT files to compute spatial water depth
        elevation_grid: numpy array of average elevation at the same grid cell size/resolution
        min_x, min_y, max_x, max_y: plot/study site bounds
        wetland_cross_sections: dictionary of linestrings (cross-sections used in plots of cross-sectional water surface elevation relative to LiDAR derived bathymetry)
        size: grid cell size in m (default=100 m)
        wetland_mask: mask to filter out poor quality data (default=mask_with_no_prior_included)
    '''
    for i, filename in enumerate(file_list):
        file = os.path.join(directory, filename)
        date = filename[29:44]
        date_format = '%Y%m%dT%H%M%S'
        date_obj = datetime.strptime(date, date_format) 
        lon, lat, east, north, h, a, area_weighted = process_transformed_swot_data(file, wetland_mask)

        df = pd.DataFrame({'height': h, 'latitude':lat, 'longitude':lon, 'area_weighted':area_weighted, 'area':a, 'east':east, 'north':north})   
        kde = gaussian_kde(df['height'], bw_method=0.5)
        x = np.linspace(min(df['height']), max(df['height']), 1000)
        kde_density = kde(x)
        threshold1 = np.max(kde_density) * 0.3
        right_tolerance1 = x[np.argmax(kde_density[np.argmax(kde_density):] < threshold1) + np.argmax(kde_density)]
        df_kde1 = df[(df['height'] <= right_tolerance1)]

        area_weighted_height = gridded_sum((min_x, min_y, max_x, max_y), size, df_kde1['east'], df_kde1['north'], df_kde1['area_weighted'])
        area_sum = gridded_sum((min_x, min_y, max_x, max_y), size, df_kde1['east'], df_kde1['north'], df_kde1['area'])

        with np.errstate(divide='ignore', invalid='ignore'):
            wse = np.true_divide(area_weighted_height, area_sum)
            wse[~np.isfinite(wse)] = np.nan

        depth = wse - elevation_grid
        depth = np.ma.masked_where(depth < 0, depth)

        height, width = depth.shape
        x_edges = np.linspace(min_x, max_x, width + 1)
        y_edges = np.linspace(min_y, max_y, height + 1)
        lon_x_edges, _ = transformer_reverse.transform(x_edges, np.full_like(x_edges, np.min(y_edges)))
        _, lat_y_edges = transformer_reverse.transform(np.full_like(y_edges, np.min(x_edges)), y_edges)

        # fig, ax = plt.subplots()
        # ax.imshow(img, extent=img_extent)
        # mesh = ax.pcolormesh(lon_x_edges, lat_y_edges, np.flipud(wse), shading='auto', cmap='Blues')
        # cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
        # cbar.set_label('Water surface elevation (m)', fontsize=12)
        # plt.xlim(xmin, xmax)
        # plt.ylim(ymin, ymax)
        # plt.xlabel('Longitude', fontsize=12)
        # plt.ylabel('Latitude', fontsize=12)
        # plt.tight_layout()
        # plt.savefig(f'{save_folder}/CL_full_gridded_wse_{str(date_obj)[0:10]}_{size}m.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
        # plt.show()

        # fig, ax = plt.subplots()
        # ax.imshow(img, extent=img_extent)
        # mesh = ax.pcolormesh(lon_x_edges, lat_y_edges, np.flipud(elevation_grid), shading='auto', cmap='terrain')
        # cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
        # cbar.set_label('Elevation (m)', fontsize=12)
        # plt.xlim(xmin, xmax)
        # plt.ylim(ymin, ymax)
        # plt.xlabel('Longitude', fontsize=12)
        # plt.ylabel('Latitude', fontsize=12)
        # plt.tight_layout()
        # plt.savefig(f'{save_folder}/CL_full_gridded_elevation_{str(date_obj)[0:10]}_{size}m.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
        # plt.show()

        fig, ax = plt.subplots()
        ax.imshow(img, extent=img_extent)
        mesh = ax.pcolormesh(lon_x_edges, lat_y_edges, np.flipud(depth), shading='auto', cmap='Blues', vmin=0, vmax=6)

        GL_volume = np.sum(~np.isnan(depth)) * size**2 * np.nanmean(depth) * 10**-6
        print(f'{filename}, {np.sum(~np.isnan(depth))*0.01} km2, {GL_volume} GL')

        if i==0:
            for wetland_name in wetland_cross_sections:
                x, y = wetland_cross_sections[wetland_name].coords.xy
                plt.plot(x, y, '-', color='black', label = f'wetland {wetland_name}', linewidth=0.7)
                ax.annotate('', xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), arrowprops=dict(arrowstyle='-|>', color='black', mutation_scale = 7, lw=0.7, shrinkA=0, shrinkB=0))

        cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label('Depth (m)', fontsize=12)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_folder}/CL_full_gridded_depth_{str(date_obj)[0:10]}_{size}m.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
        plt.show()

        np.save(f'{save_folder}/elevation_grid_{str(date_obj)[0:10]}_{size}m.npy', np.flipud(elevation_grid))
        np.save(f'{save_folder}/water_height_{str(date_obj)[0:10]}_{size}m.npy', np.flipud(wse))
        np.save(f'{save_folder}/depth_{str(date_obj)[0:10]}_{size}m.npy', np.flipud(depth).filled(np.nan))
        np.save(f'{save_folder}/lon_x_edges_{size}m.npy', lon_x_edges)
        np.save(f'{save_folder}/lat_y_edges_{size}m.npy', lat_y_edges)
    return None

#######################################
##  Cross-sectional wetland profiles ##
#######################################

def get_laz_bounding_box(las_file):
    '''
    Extracts the bounding box of LiDAR point cloud files from the file header.
    
    Input: 
        las_file: file name
    
    Returns: shapely box defined by the minimum and maximum x and y bounds'''
    las = laspy.open(las_file)
    min_x, min_y = las.header.min[0], las.header.min[1]
    max_x, max_y = las.header.max[0], las.header.max[1]
    return box(min_x, min_y, max_x, max_y) 

def point_cloud_to_df(las_directory, polygon, output_file):
    '''
    Creates a dataframe of ground classified LiDAR point cloud data (x,y,z) within a given polygon and save to a .parquet file
    
    Inputs:
        las_directory: directory containing LiDAR point cloud files (.laz)
        polygon: shapely polygon defining the area from which to extract LiDAR points 
        output_file: full output filename 
        '''
    transformed_polygon = Polygon([transformer.transform(lon, lat) for lat, lon in polygon.exterior.coords])

    las_files = [os.path.join(las_directory, f) for f in os.listdir(las_directory) if f.endswith(".laz")]
    intersecting_files = []
    for las_file_path in las_files:
        bbox = get_laz_bounding_box(las_file_path)
        if transformed_polygon.intersects(bbox):
            intersecting_files.append(las_file_path)
    
    df = pd.DataFrame(columns=['x', 'y', 'z', 'classification'])
    for las_file_path in las_files:
        bbox = get_laz_bounding_box(las_file_path)
        if transformed_polygon.intersects(bbox):
            las = laspy.read(las_file_path)
            x = np.array(las.x)
            y = np.array(las.y)
            z = np.array(las.z)
            classification = np.array(las.classification)
    
            point_cloud_gdf = pd.DataFrame({'x': x, 'y': y, 'z': z, 'classification': classification})
            point_cloud_gdf = point_cloud_gdf[point_cloud_gdf['classification']==2]
            df = pd.concat([df, point_cloud_gdf], ignore_index=True)
            # print(len(df))

    save_df = df.drop('classification', axis=1)
    save_df.to_parquet(output_file, compression='snappy')
    return None

def extract_elevation_transect(wetland_cross_sections, wetland_name_list, lidar_file, buffer_radius=200):
    '''
    Extracts elevation/bathymetry values at a particular interval for a series of wetland cross-sections. The transect intervals and values are saved to a .npy file.
    
    Inputs: 
        wetland_cross_sections: linestring dictionary of transects to show wetland profiles
        wetland_name_list: list of wetland names (as in the linestring dictionary) for which point cloud data is in the LiDAR file
        lidar_file: file name of .parquet file containing LiDAR point data along the full buffered transect area.  
        buffer_radius: sampling radius of points to calculate an average value from (default=200 m)
    '''
    df = pd.read_parquet(lidar_file)

    for wetland_name in wetland_name_list:
        line = transform(transformer.transform, wetland_cross_sections[wetland_name])
        buffer = line.buffer(buffer_radius + 50)

        mask = vectorized.contains(buffer, df['x'].astype(float), df['y'].astype(float))
        lidar_df = df[mask]

        distances = np.arange(0, line.length + 1, buffer_radius)
        sample_points = [line.interpolate(d) for d in distances] 
        elevation = []  
        site_data = []
        for pt in sample_points:
            dx = lidar_df['x'] - pt.x
            dy = lidar_df['y'] - pt.y
            distances = np.sqrt(dx**2 + dy**2)
            inside_mask = distances <= buffer_radius
            if np.any(inside_mask):
                mean_val = lidar_df['z'][inside_mask].mean()
            else:
                mean_val = np.nan
            elevation.append(mean_val)

        cum_distances = [line.project(pt) for pt in sample_points]
        site_data.append((cum_distances, elevation))
        np.save(f'{save_folder}/explicit_interval_adjusted_transect_{buffer_radius}_{wetland_name}.npy', np.array(site_data))
    return None

def cross_sectional_wse_topography(linestring_dictionary, file_list, wetland_mask, buffer_radius=200):
    '''
    Calculates and plots the average area-weighted water surface elevation of a series of wetland cross-sections relative to the bathymetry for a given list 
    of SWOT files. SWOT data from each date is saved to a .pkl file labelled by date and wetland name. The final plot showing the cross-sections for all 
    dates in the file_list relative to the bathymetry is saved to a .png file. 
    
    Inputs: 
        linestring_dictionary: dictionary of linestrings for a series of wetlands to calculate cross-sectional water surface elevation along
        file_list: list of SWOT files to calculate and plot cross-sectional water surface elevation for
        wetland_mask: mask to filter out poor quality data (default=mask_with_no_prior_included)
        buffer_radius: sampling radius of points to calculate an average value from (default=200 m)
    '''
    plt.figure(figsize=(18,5))
    gap = 500

    tick_positions = []
    tick_labels = []
    x_offset = 0
    
    site_data=[]
    for i, wetland_name in enumerate(linestring_dictionary):
        elevation_data = np.load(f'{save_folder}/explicit_interval_adjusted_transect_{buffer_radius}_{wetland_name}.npy')
        cross_section_distance = np.max(elevation_data[0,0])
        label = 'Bathymetry' if i==0 else None
        site_data.append((elevation_data[0,0], elevation_data[0,1], cross_section_distance))

    for i, (x, y, dur) in enumerate(site_data):
        x_shifted = np.asarray(x) + x_offset
        plt.fill_between(x_shifted, y, 0, color='white', zorder=2) 
        plt.plot(x_shifted, y, linewidth=1.5, color = 'black', label='Bathymetry' if i==0 else None, zorder=3)
        
        tick_intervals = np.arange(0, dur, 2000)
        for t in tick_intervals:
            tick_positions.append(x_offset + t)
            tick_labels.append(f"{int(t/1000)}")

        x_offset += dur + gap
        plt.axvline(x=x_offset - gap/2, color='gray', linestyle=':', linewidth=2) # Draw dotted vertical separators

    for i, filename in enumerate(file_list):
        colors = colour_sequence(len(file_list), cmap_name='coolwarm')
        time_colour = colors[i]
        file = os.path.join(directory, filename)
        print(filename)
        date = filename[29:44]
        date_format = '%Y%m%dT%H%M%S'
        date_obj = datetime.strptime(date, date_format) 
        lon, lat, east, north, h, a, area_weighted = process_transformed_swot_data(file, wetland_mask)

        site_data = []
        for i, wetland_name in enumerate(linestring_dictionary):
            label = str(date_obj)[0:10] if i==0 else None
            line = transform(transformer.transform, linestring_dictionary[wetland_name])
            distances = np.arange(0, line.length + 1, buffer_radius)
            sample_points = [line.interpolate(d) for d in distances]
    
            wse = []  
            for pt in sample_points:
                dx = east - pt.x
                dy = north - pt.y
                distances = np.sqrt(dx**2 + dy**2)
                inside_mask = distances <= buffer_radius
 
                if np.any(inside_mask):
                    height_area_sum = area_weighted[inside_mask].sum()
                    area_sum = a[inside_mask].sum()
                    mean_val = height_area_sum/area_sum
                else:
                    mean_val = np.nan
                wse.append(mean_val)
        
            cum_distances = [line.project(pt) for pt in sample_points]
            cross_section_distance = np.max(cum_distances)
            site_data.append((cum_distances, wse, cross_section_distance))
            
            with open(f'{save_folder}/site_data_{str(date_obj)[0:10]}_{wetland_name}.pkl', "wb") as f:
                pickle.dump(site_data, f)

            with open(f'{save_folder}/site_data_{str(date_obj)[0:10]}_{wetland_name}.pkl', "rb") as f:
                site_data = pickle.load(f)

            x_offset = 0  
            for i, (x, y, dur) in enumerate(site_data):
                x_shifted = np.asarray(x) + x_offset
                plt.plot(x_shifted, y, linewidth=3, color = time_colour, label=label, zorder=1)
                x_offset += dur + gap

    plt.xlabel('Distance along each wetland transect (km)', fontsize=18)
    plt.ylabel('Water surface elevation (m)', fontsize=18)
    plt.xticks(tick_positions, tick_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-250, x_offset-250)
    plt.ylim(25, 32)
    
    handles, labels = plt.gca().get_legend_handles_labels() # Get handles and labels
    plt.legend(handles[::-1], labels[::-1], loc="upper right", bbox_to_anchor=(1.17, 1), fontsize=18) # Draw legend with reversed order
    plt.tight_layout(pad=0.1)
    plt.savefig(f'{save_folder}/fig1_cross_section_{str(date_obj)[0:4]}.png', bbox_inches='tight', pad_inches=0.2, dpi=900)
    plt.show()
    return None

def cross_sectional_wse_topography_from_saved(linestring_dictionary, file_list, buffer_radius=200):
    '''
    Plots the average area-weighted water surface elevation of a series of wetland cross-sections relative to the bathymetry for a given list 
    of SWOT files, using the saved .pkl file labelled by the date and wetland name. The final plot showing the cross-sections for all 
    dates in the file_list relative to the bathymetry is saved to a .png file. 
    
    Inputs: 
        linestring_dictionary: dictionary of linestrings for a series of wetlands to calculate cross-sectional water surface elevation along
        file_list: list of SWOT files to calculate and plot cross-sectional water surface elevation for
        buffer_radius: sampling radius of points to calculate an average value from (default=200 m)
    '''
    plt.figure(figsize=(18,5))
    gap = 500
    tick_positions = []
    tick_labels = []
    x_offset = 0
    
    site_data=[]
    for i, wetland_name in enumerate(linestring_dictionary):
        elevation_data = np.load(f'{save_folder}/explicit_interval_adjusted_transect_{buffer_radius}_{wetland_name}.npy')
        cross_section_distance = np.max(elevation_data[0,0])
        label = 'Bathymetry' if i==0 else None
        site_data.append((elevation_data[0,0], elevation_data[0,1], cross_section_distance))

    for i, (x, y, dur) in enumerate(site_data):
        x_shifted = np.asarray(x) + x_offset
        plt.fill_between(x_shifted, y, 0, color='white', zorder=2) 
        plt.plot(x_shifted, y, linewidth=1.5, color = 'black', label='Bathymetry' if i==0 else None, zorder=3)
        
        tick_intervals = np.arange(0, dur, 2000)
        for t in tick_intervals:
            tick_positions.append(x_offset + t)
            tick_labels.append(f"{int(t/1000)}")

        x_offset += dur + gap # Increase offset
        plt.axvline(x=x_offset - gap/2, color='gray', linestyle=':', linewidth=2) # Draw dotted vertical separators

    for i, filename in enumerate(file_list):
        colors = colour_sequence(len(file_list), cmap_name='coolwarm')
        time_colour = colors[i]
        print(filename)
        date = filename[29:44]
        date_format = '%Y%m%dT%H%M%S'
        date_obj = datetime.strptime(date, date_format) 

        with open(f'{save_folder}/site_data_{str(date_obj)[0:10]}_{wetland_name}.pkl', "rb") as f:
            site_data = pickle.load(f)
        x_offset = 0  
        for i, (x, y, dur) in enumerate(site_data):
            label = str(date_obj)[0:10] if i==0 else None
            x_shifted = np.asarray(x) + x_offset
            plt.plot(x_shifted, y, linewidth=3, color = time_colour, label=label, zorder=1)
            x_offset += dur + gap # Increase offset
    
    plt.xlabel('Distance along each wetland transect (km)', fontsize=18)
    plt.ylabel('Water surface elevation (m)', fontsize=18)
    plt.xticks(tick_positions, tick_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-250, x_offset-250)
    plt.ylim(25, 32)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc="upper right", bbox_to_anchor=(1.17, 1), fontsize=18) # Draw legend with reversed order
    plt.tight_layout(pad=0.1)
    plt.savefig(f'{save_folder}/fig1_cross_section_{str(date_obj)[0:4]}_includingnoprior.png', bbox_inches='tight', pad_inches=0.2, dpi=900)
    plt.show()
    return None

##################################################
## Wetland Water Surface Elevation Time Series  ##
##################################################

def extract_wetland_pixc_wse(directory, wetland_polygons, VerC_mask, VerD_mask):
    '''
    Reads in all parameters, extracts values, masks and filters the values to remove poor quality data and create dataframes of water surface elevation and standard deviation 
    (all pixels and retained pixels after KDE filtering) with corresponding dates from PIXC data files. When there are two SWOT files with the same date, these 
    files are merged before undertaking masking, filtering and calculation of water surface elevation and standard deviations. 
 
    Inputs:
        directory: Directory containing pixel cloud netCDF files
        wetland_polygons: dictionary of wetland polygons 
        VerC_mask: mask for SWOT PIXC Version C data 
        VerD_mask: mask for SWOT PIXC Version D data 
    Returns: 
        wetland_dataframes: list of dataframes (in order of wetlands in wetland_polygon dictionary) containing the date of SWOT data, area-weighted average water surface elevation standard deviation, 
                            area-weighted water surface elevation with KDE filtering and standard deviation of KDE filtered pixel water surface elevations  

    '''
    wetland_dataframes = [pd.DataFrame(columns=["Date", "WSE", "ST_DEV", "WSE_KDE", "ST_DEV_KDE"]) for _ in range(len(wetland_polygons))]
    wetland_bounds = get_wetland_bounding_box(wetland_polygons)

    files = os.listdir(directory)
    i=0
    while i < len(files):
        filename = files[i]
        f = os.path.join(directory, filename)
        date = filename[29:44]
        date_format = '%Y%m%dT%H%M%S'
        date_obj = datetime.strptime(date, date_format)
        ds1 = nc.Dataset(f)
        data = read_pixc_values_tide(ds1)
 
        if i+1 < len(files):
            if filename[29:37] in files[i+1]:
                study_site_mask = (data[1] > ymin) & (data[1] < ymax) & (data[2] > xmin) & (data[2] < xmax)
                filtered_arrays = [array[study_site_mask] for array in data]
                f = os.path.join(directory, files[i+1])
                ds1 = nc.Dataset(f)
                data1 = read_pixc_values_tide(ds1)
                data = [np.hstack((a, b)) for a, b in zip(filtered_arrays, data1)]
                i+=2
            else:
                i+=1
        else:
            i+=1
    
        for wetland_poly, wetland_df, bounds in zip(wetland_polygons, wetland_dataframes, wetland_bounds):
            study_site_mask = (data[1] > bounds[2]) & (data[1] < bounds[3]) & (data[2] > bounds[0]) & (data[2] < bounds[1])
            filtered_arrays = [array[study_site_mask] for array in data]
            [height, latitude, longitude, classification, sigma0, geoid, intf_qual, geolocation_qual, pixel_area, classification_qual, sig0_qual, solid_earth_tide, load_tide_fes, load_tide_got, pole_tide] = filtered_arrays

            ### Create mask to remove bad data ###
            if 'PIC' in f or 'PGC' in f:
                ### Create mask to remove bad data ###
                mask = VerC_mask(classification, sigma0, intf_qual, geolocation_qual, classification_qual, sig0_qual)
            elif 'PID' in f:
                mask = VerD_mask(classification, sigma0, intf_qual, geolocation_qual, classification_qual, sig0_qual)
 
            ### Apply mask and tidal models: Note that load_tide_fes is used, not load_tide_got ###
            Masked_Pixels = (height - geoid - solid_earth_tide - load_tide_fes - pole_tide) * mask 
            Masked_Height = Masked_Pixels[~np.isnan(Masked_Pixels)]
            Masked_pixel_area = pixel_area[~np.isnan(Masked_Pixels)]
            latitude = latitude[~np.isnan(Masked_Pixels)]
            longitude = longitude[~np.isnan(Masked_Pixels)]
 
            ### Extract values that are within the polygon and append to new arrays ###
            points = shapely.points(longitude, latitude)
            tree = STRtree(points)
            indices = tree.query(wetland_polygons[wetland_poly], predicate='contains')
           
            if not len(indices) >= 5:
                # print('Insufficient valid data in polygon')
                continue
 
            Good_Pixel_Heights = Masked_Height[indices].compressed()
            Good_Pixel_Areas = Masked_pixel_area[indices].compressed()
 
            if len(Good_Pixel_Heights[~np.isnan(Good_Pixel_Heights)]) > 50:
                if len(Good_Pixel_Heights[~np.isnan(Good_Pixel_Heights)]) > 1000:
                    kde = gaussian_kde(Good_Pixel_Heights, bw_method=0.05)
                elif (len(Good_Pixel_Heights[~np.isnan(Good_Pixel_Heights)]) <= 1000) and (len(Good_Pixel_Heights[~np.isnan(Good_Pixel_Heights)]) > 100):
                    kde = gaussian_kde(Good_Pixel_Heights, bw_method=0.1)
                elif (len(Good_Pixel_Heights[~np.isnan(Good_Pixel_Heights)]) <= 100) and (len(Good_Pixel_Heights[~np.isnan(Good_Pixel_Heights)]) > 50):
                    kde = gaussian_kde(Good_Pixel_Heights, bw_method=0.2)
                
                x = np.linspace(np.nanmin(Good_Pixel_Heights), np.nanmax(Good_Pixel_Heights), 1000)
                kde_density = kde(x)
                threshold = np.max(kde_density) * 0.2 # Define tolerance threshold: Look for where the density drops 20% from the peak
                left_tolerance = x[np.argmax(kde_density) - np.argmax((kde_density[:np.argmax(kde_density)] < threshold)[::-1]) - 1]
                right_tolerance = x[np.argmax(kde_density[np.argmax(kde_density):] < threshold) + np.argmax(kde_density)]
                KDE_Pixel_Heights = Good_Pixel_Heights[np.where((Good_Pixel_Heights >= left_tolerance) & (Good_Pixel_Heights <= right_tolerance))]
                KDE_Pixel_Areas = Good_Pixel_Areas[np.where((Good_Pixel_Heights >= left_tolerance) & (Good_Pixel_Heights <= right_tolerance))]
            else:
                KDE_Pixel_Heights = Good_Pixel_Heights
                KDE_Pixel_Areas = Good_Pixel_Areas
 
            if not len(Good_Pixel_Heights[~np.isnan(Good_Pixel_Heights)]) >= 5:
                # print('Insufficient valid data')
                continue
 
            ### Calculate mean WSE weighted by pixel area ###
            Good_Mean = (np.nansum(Good_Pixel_Heights*Good_Pixel_Areas))/(np.nansum(Good_Pixel_Areas))
            KDE_Mean = (np.nansum(KDE_Pixel_Heights*KDE_Pixel_Areas))/(np.nansum(KDE_Pixel_Areas))
 
            ### Append valid data to arrays ###
            if math.isnan(Good_Mean) == False:
                wetland_df.loc[len(wetland_df)] = [date_obj, Good_Mean, np.nanstd(Good_Pixel_Heights), KDE_Mean, np.nanstd(KDE_Pixel_Heights)]
    return wetland_dataframes

def wetland_time_series(directory, wetland_polygons, VerC_mask, VerD_mask, fstring_save_to_file):
    '''
    Creates a series of dataframes of water surface elevation and standard deviation over time for SWOT data files in a given directory and save each dataframe to a .parquet file containing the wetland polygon name in the file name. 
    
    Inputs:
        directory: Directory containing pixel cloud netCDF files
        wetland_polygons: dictionary of wetland polygons 
        VerC_mask: mask for SWOT PIXC Version C data 
        VerD_mask: mask for SWOT PIXC Version D data 
        fstring_save_to_file: string to be included in the file name that the dataframes will be saved to
    '''
    wetland_dataframes = extract_wetland_pixc_wse(directory, wetland_polygons, VerC_mask, VerD_mask)
    for wetland_poly, wetland_df in zip(wetland_polygons, wetland_dataframes):
        wetland_df.to_parquet(f'{save_folder}/{fstring_save_to_file}_{wetland_poly}.parquet')
    return None

def plot_time_series(wetland_list, gauge_list, new_mask_filename, maubant_mask_filename, wrmse=True, maubant=True, label='In situ gauge', threshold_days=1):
    '''
    Plots the water level time series from the new mask and the Maubant et al. (2025) mask against a gauge time series and saves the outputs for each wetland and gauge. 
    The absolute bias of the SWOT values (typically arising from the SWOT values representing geoid height, while the gauges represent water level) is subtracted and the WRMSE of 
    the adjusted SWOT water surface elevation values relative to the gauge is computed. Each of these plots are saved to a .png file with the wetland polygon name noted in the file name. 

    Inputs:
        wetland_list: list of wetlands to compare and plot against the Maubant et al. (2025) mask and gauge
        gauge_list: corresponding list of gauges to plot the wetland time series against
        new_mask_filename: fstring_save_to_file defined in the wetland_time_series function to save the time series based on the new mask to
        maubant_mask_filename: fstring_save_to_file defined in the wetland_time_series function to save the time series based on the Maubant et al. (2025) mask to
        wrmse: whether to compute WRMSE or not (default=True, i.e. the WRMSE is computed)
        maubant: whether to plot the time series masked using the Maubant et al. (2025) approach or not
        label: label for the gauge file (default='In situ gauge')
        threshold_days: maximum number of days between the SWOT and gauge data to compare a SWOT data point with the closest gauge water level for computing the absolute bias (default=1 day)
    '''

    for wetland_poly, gauge in zip(wetland_list, gauge_list):
        gauge = pd.read_csv(gauge, sep='\t', skiprows=9)
        swot_time_series = {}
        swot_time_series['New mask'] = pd.read_parquet(f'{save_folder}/{new_mask_filename}_{wetland_poly}.parquet')
        swot_time_series['Maubant et al. (2025) mask'] = pd.read_parquet(f'{save_folder}/{maubant_mask_filename}_{wetland_poly}.parquet')
        colours = colour_sequence(len(swot_time_series), 'coolwarm_r')

        gauge = gauge.copy()
        gauge['Date'] = pd.to_datetime(gauge.iloc[:,0]).dt.tz_localize(None)
        results = {}
    
        plt.figure(figsize=(12, 4))
        plt.plot(gauge['Date'], gauge.iloc[:,1], 'o-', label=label, color='black', markersize=2, zorder=1)
    
        for i, (mask_name, df_b) in enumerate(swot_time_series.items()):
            df_b = df_b.copy()
            df_b['Date'] = pd.to_datetime(df_b.iloc[:,0]).dt.tz_localize(None)
            matched_rows = []
    
            for idx_b, row_b in df_b.iterrows():
                date_b = row_b['Date']
                # Find matches in gauge within threshold
                close_dates = gauge[(gauge['Date'] - date_b).abs() <= pd.Timedelta(days=threshold_days)]
                if not close_dates.empty:
                    closest_idx = (close_dates['Date'] - date_b).abs().idxmin() # closest match
                    row_a = gauge.loc[closest_idx]
    
                    if i==0:
                        matched_rows.append({'Date_b': row_b['Date'], 'WSE': row_b['WSE_KDE'], 'ST_DEV': row_b['ST_DEV_KDE'], 'Date_a': row_a['Date'], 'observed': row_a.iloc[1]})
                    elif i==1:
                        matched_rows.append({'Date_b': row_b['Date'], 'WSE': row_b['WSE'], 'ST_DEV': row_b['ST_DEV'], 'Date_a': row_a['Date'], 'observed': row_a.iloc[1]})

            if len(matched_rows) < 1:
                continue
            df_matched = pd.DataFrame(matched_rows)
            df_matched.dropna(inplace=True)
            df_matched = df_matched[df_matched['ST_DEV'] != 0]
    
            # Bias adjustment
            bias = np.nanmean(df_matched['WSE'] - df_matched['observed'])
            df_matched['WSE_adj'] = df_matched['WSE'] - bias
    
            # Weighted RMSE
            df_matched['weight'] = 1 / (df_matched['ST_DEV'] ** 2)
            squared_error = (df_matched['WSE_adj'] - df_matched['observed']) ** 2
            weighted_rmse = np.sqrt(np.sum(df_matched['weight'] * squared_error) / np.sum(df_matched['weight']))
            rmse = np.sqrt(np.sum(squared_error)/len(squared_error))
            results[mask_name] = {'bias': bias, 'weighted_rmse': weighted_rmse, 'rmse': rmse, 'matched': df_matched}
            if i==0:
                plt.errorbar(df_b['Date'], df_b['WSE_KDE'] - bias, yerr=df_b['ST_DEV_KDE'], fmt='o', capsize=2, label=f'{mask_name}: WRMSE ({weighted_rmse:.2f} m)' if wrmse is True else f'{mask_name}', color=colours[i])
            elif i==1:
                if maubant==True:
                    plt.errorbar(df_b['Date'], df_b['WSE'] - bias, yerr=df_b['ST_DEV'], fmt='o', capsize=2, label=f'{mask_name}: WRMSE ({weighted_rmse:.2f} m)'if wrmse is True else f'{mask_name}', color=colours[i])
                else:
                    continue
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Water level (m)', fontsize=15)
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_folder}/timeseries_{wetland_poly}_{mask_name}.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
        plt.show()
    return None

def plot_all_time_series_gauge_discharge(gauge_filename, discharge_filename, wetland_polygons, timeseries_filename, threshold_days=1, connecting_threshold=21):
    '''
    Plots the wetland polygons over the study site, plot the water level time series of all wetlands (adjusted for absolute bias relative to the plotted gauge) with a nearby 
    water level gauge and also plot the discharge rate from another gauge in a subplot alongside the wetland time series and water level gauge. The plot of wetland polygon locations within 
    the study site and the time series plot are saved to .png files. 

    Inputs:
        gauge_filename: file name for gauge with water level measurements (file in format used by Australian Bureau of Meteorology (BOM) .txt files)
        discharge_filename: file name for gauge with discharge rate measurements in cumecs (file in format used by Australian Bureau of Meteorology (BOM) .txt files)
        wetland_polygons: dictionary of wetland polygons
        maubant_mask_filename: time series file name (fstring_save_to_file parameter defined when running the wetland_time_series function)
        threshold_days: maximum number of days between the SWOT and gauge data to compare a SWOT data point with the closest gauge water level for computing the absolute bias (default=1 day)
        connecting_threshold: maximum number of days for which to connect consecutive points in a time series for each wetland (default=21 days)
    '''
    date_list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        ds1 = nc.Dataset(f)
        swot_polygon = Polygon([(getattr(ds1, 'inner_first_longitude'), getattr(ds1, 'inner_first_latitude')), (getattr(ds1, 'inner_last_longitude'), getattr(ds1, 'inner_last_latitude')), (getattr(ds1, 'outer_last_longitude'), getattr(ds1, 'outer_last_latitude')), (getattr(ds1, 'outer_first_longitude'), getattr(ds1, 'outer_first_latitude'))])
        if swot_polygon.intersects(box(xmin, ymin, xmax, ymax)):
            date = filename[29:44]
            date_format = '%Y%m%dT%H%M%S'
            date_obj = datetime.strptime(date, date_format)
            date_list.append(date_obj)

    colors = ["#26b710",'#d95f02', "#3a6ce2",  '#e6ab02', "#24ada4", '#e7298a', '#a6761d', "#98BF50"]
    
    fig, ax = plt.subplots()
    ax.imshow(img, extent=img_extent)
    
    colours = {}
    swot_time_series = {}
    for i, wetland_poly in enumerate(wetland_polygons):
        masked = pd.read_parquet(f'{save_folder}/{timeseries_filename}_{wetland_poly}.parquet')
        swot_time_series[wetland_poly] = masked
        colours[wetland_poly] = colors[i]
        x, y = wetland_polygons[wetland_poly].exterior.xy
        plt.plot(x,y, color=colors[i], label=wetland_poly)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Longitude', fontsize=11)
    plt.ylabel('Latitude', fontsize=11)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend(loc='upper right', ncol=2, fontsize=10)
    plt.savefig(f'{save_folder}/BM_study_site_with_wetland_polygons.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.show()

    gauge = pd.read_csv(gauge_filename, sep = '\t', skiprows=9)
    gauge = gauge.copy()
    gauge['Date'] = pd.to_datetime(gauge.iloc[:,0]).dt.tz_localize(None)
   
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True, constrained_layout=True)
    ax = ax.flatten()

    discharge = pd.read_csv(discharge_filename, sep = '\t', skiprows=9)
    discharge['Date'] = pd.to_datetime(discharge.iloc[:,0]).dt.tz_localize(None)
    ax[1].plot(discharge['Date'], discharge.iloc[:,1]*0.0864, 'o-', label='Yarrawonga Weir Gauge', color='black', markersize=3)
    ax[1].legend()
 
    label_list = []
    for mask_name, swot_df in swot_time_series.items():
        swot_df = swot_df.copy()
        swot_df['Date'] = pd.to_datetime(swot_df.iloc[:,0]).dt.tz_localize(None)
        matched_rows = []
 
        for idx_b, row_b in swot_df.iterrows():
            date_b = row_b['Date']
            # Find matches in gauge within threshold
            within_threshold = gauge[(gauge['Date'] - date_b).abs() <= pd.Timedelta(days=threshold_days)]
            if not within_threshold.empty:
                # Get closest match
                closest_idx = (within_threshold['Date'] - date_b).abs().idxmin()
                row_a = gauge.loc[closest_idx]
                matched_rows.append({'Date_b': row_b['Date'], 'WSE': row_b['WSE_KDE'], 'ST_DEV': row_b['ST_DEV_KDE'], 'Date_a': row_a['Date'], 'observed': row_a.iloc[1]})
 
        df_matched = pd.DataFrame(matched_rows)
        df_matched.dropna(inplace=True)
        df_matched = df_matched[df_matched['ST_DEV'] != 0]
 
        # Bias adjustment
        bias = np.nanmean(df_matched['WSE'] - df_matched['observed'])
        swot_df['WSE_adj'] = swot_df['WSE_KDE'] - bias
 
        threshold = pd.Timedelta(days=connecting_threshold)
        for i in range(len(swot_df['Date']) - 1):
            if abs(swot_df['Date'][i + 1] - swot_df['Date'][i]) <= threshold:
                line = ax[0].errorbar([swot_df['Date'][i], swot_df['Date'][i+1]], [swot_df['WSE_adj'][i], swot_df['WSE_adj'][i+1]], fmt='-o', markersize=2, capsize=0, alpha=0.7, color=colours[mask_name])
                ax[0].fill_between([swot_df['Date'][i], swot_df['Date'][i+1]], [swot_df['WSE_adj'][i]-swot_df['ST_DEV_KDE'][i], swot_df['WSE_adj'][i+1] - swot_df['ST_DEV_KDE'][i+1]], [swot_df['WSE_adj'][i] + swot_df['ST_DEV_KDE'][i], swot_df['WSE_adj'][i+1] + swot_df['ST_DEV_KDE'][i+1]], alpha=0.2, color=colours[mask_name])
            else:
                line = ax[0].errorbar(swot_df['Date'][i], swot_df['WSE_adj'][i], fmt='-o', markersize=2, capsize=0, alpha=0.7, color=colours[mask_name])
            if mask_name not in label_list:
                line.set_label(mask_name)
                label_list.append(mask_name)
 
    ax[0].plot(date_list, [0.4] * len(date_list), 'o', color='blue', markersize=3, zorder=10)
    ax[0].spines['bottom'].set_position(('data', 0.4))
    ax[0].spines['bottom'].set_color('black')
    ax[0].spines['left'].set_bounds(0.4, ax[0].get_ylim()[1])
    ax[0].spines['right'].set_bounds(0.4, ax[0].get_ylim()[1])
    ax[0].set_ylabel('Water level (m)')
    ax[0].tick_params(axis='x', length=8)
    plt.xlabel('Date')
    plt.ylabel('Discharge rate (GL/day)')
    plt.xlim(np.min(gauge['Date'])-timedelta(days=10), np.max(gauge['Date'])+timedelta(days=10))
    plt.tight_layout()
    plt.savefig(f'{save_folder}/timeseries_gauge.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.show()
 
    return None

#####################################
## Spatial Gridding                ##
#####################################

def gridded_wetland_height_anomaly(directory, file_list, wetland_polygons, mask_function):
    '''
    Calculates and plots gridded water surface elevation and gridded water surface elevation anomalies (relative to the mean gridded water surface elevation) for a given list of files and wetland polygons. 
    Histograms of pixel level and grid level water surface elevation are also plotted. All plots are saved to .png files using the file date, wetland name and plot type. 

    Inputs: 
        directory: directory containing SWOT files
        file_list: list of SWOT files
        wetland_polygons: dictionary of wetland polygons
        mask_function: mask_function to be used to filter out poor quality data
    '''
    wetland_bounds = get_wetland_bounding_box(wetland_polygons)

    for filename in file_list:
        f = os.path.join(directory, filename)
        print('File:', filename)
        date = filename[29:44]
        date_format = '%Y%m%dT%H%M%S'
        date_obj = datetime.strptime(date, date_format)        
        ds1 = nc.Dataset(f)
        swot_polygon = Polygon([(getattr(ds1, 'inner_first_longitude'), getattr(ds1, 'inner_first_latitude')), (getattr(ds1, 'inner_last_longitude'), getattr(ds1, 'inner_last_latitude')), (getattr(ds1, 'outer_last_longitude'), getattr(ds1, 'outer_last_latitude')), (getattr(ds1, 'outer_first_longitude'), getattr(ds1, 'outer_first_latitude'))])
        data = read_pixc_values_tide(ds1) # Call function to read in data from SWOT PIXC file

        for i, (wetland_poly, bounds) in enumerate(zip(wetland_polygons, wetland_bounds)):
            if not swot_polygon.contains(wetland_polygons[wetland_poly]):
                # print(f'Insufficient wetland polygon coverage: skipping {wetland_poly}')
                continue
            study_site_mask = (data[1] > bounds[2]) & (data[1] < bounds[3]) & (data[2] > bounds[0]) & (data[2] < bounds[1])
            filtered_arrays = [array[study_site_mask] for array in data]
            [height, latitude, longitude, classification, sigma0, geoid, intf_qual, geolocation_qual, pixel_area, classification_qual, sig0_qual, solid_earth_tide, load_tide_fes, load_tide_got, pole_tide] = filtered_arrays

            mask = mask_function(classification, sigma0, intf_qual, geolocation_qual, classification_qual, sig0_qual)
            Masked_Pixels = (height - geoid - solid_earth_tide - load_tide_fes - pole_tide) * mask # Apply mask and tidal models to obtain water height relative to the EGM2008 geoid for pixels with good quality data
            Masked_Height = Masked_Pixels[~np.isnan(Masked_Pixels)]
            Masked_pixel_area = pixel_area[~np.isnan(Masked_Pixels)]
            latitude = latitude[~np.isnan(Masked_Pixels)]
            longitude = longitude[~np.isnan(Masked_Pixels)]

            if len(latitude) > 0:
                easting, northing = transformer.transform(longitude, latitude)
                polygon = transform(transformer.transform, wetland_polygons[wetland_poly])
                df = pd.DataFrame({'x': easting, 'y': northing, 'height': Masked_Height, 'area_weighted': Masked_Height*Masked_pixel_area, 'area':Masked_pixel_area, 'Longitude': longitude, 'Latitude':latitude})
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
                polygon_gdf = df[gdf.geometry.intersects(polygon)]
                polygon_df = polygon_gdf.copy()
                x,y = polygon.exterior.xy
                grid_bounds = (np.min(x), np.min(y), np.max(x), np.max(y)) 
                min_x, min_y, max_x, max_y = np.min(x), np.min(y), np.max(x), np.max(y)

                if len(polygon_df) > 50:
                    if len(polygon_df) > 1000:
                        kde = gaussian_kde(polygon_df['height'], bw_method=0.05)
                    elif (len(polygon_df) <=1000) and (len(polygon_df) > 100):
                        kde = gaussian_kde(polygon_df['height'], bw_method=0.1)
                    elif (len(polygon_df) <= 100) and (len(polygon_df) > 50):
                        kde = gaussian_kde(polygon_df['height'], bw_method=0.2)
                    
                    x = np.linspace(min(polygon_df['height']), max(polygon_df['height']), 1000)
                    kde_density = kde(x)
                    threshold1 = np.max(kde_density) * 0.2
                    left_tolerance1 = x[np.argmax(kde_density) - np.argmax((kde_density[:np.argmax(kde_density)] < threshold1)[::-1]) - 1]
                    right_tolerance1 = x[np.argmax(kde_density[np.argmax(kde_density):] < threshold1) + np.argmax(kde_density)]
                    filtered_df = polygon_df[(polygon_df['height'] >= left_tolerance1) & (polygon_df['height'] <= right_tolerance1)] 
                else:
                    filtered_df = polygon_df
                
                if len(filtered_df) > 5:
                    for size in [100]:                        
                        filtered_x = filtered_df['x'].values
                        filtered_y = filtered_df['y'].values
                        filtered_area_weighted_height = filtered_df['area_weighted'].values
                        filtered_area = filtered_df['area'].values

                        pixel_mean = np.sum(filtered_area_weighted_height)/np.sum(filtered_area)
                        pixel_height_anomaly = filtered_df['height'].values - pixel_mean
                        
                        fig, ax = plt.subplots()
                        ax.imshow(img, extent=img_extent) # Extent matched to actual coordinates
                        sc = plt.scatter(filtered_df['Longitude'], filtered_df['Latitude'], c=pixel_height_anomaly, cmap=blue_white_green_cmap, s=0.5, vmin=-0.25, vmax=0.25)
                        cbar=plt.colorbar(sc)
                        cbar.set_label(label='Pixel water surface elevation anomaly (m)', fontsize=11)
                        ax.ticklabel_format(useOffset=False)
                        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                        plt.xlim(bounds[0]-0.001, bounds[1]+0.002)
                        plt.ylim(bounds[2]-0.001, bounds[3]+0.003)
                        plt.xlabel('Longitude', fontsize=11)
                        plt.ylabel('Latitude', fontsize=11)
                        plt.savefig(f'{save_folder}/spatial_pixel_height_anomaly_{wetland_poly}_{str(date_obj)[0:10]}.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
                        plt.show()     

                        fig, ax = plt.subplots(1,2, figsize=(10,2), constrained_layout=True)
                        counts, bins, patches = ax[0].hist(pixel_height_anomaly, bins=np.arange(-0.3, 0.3, 0.01), edgecolor='black')
                        ax[0].set_ylabel('Pixel frequency', fontsize=11)

                        bin_centers = 0.5 * (bins[:-1] + bins[1:])
                        cmap = plt.get_cmap(blue_white_green_cmap)
                        norm = Normalize(vmin=-0.2, vmax=0.2)

                        for center, patch in zip(bin_centers, patches):
                            color = cmap(norm(center))
                            patch.set_facecolor(color)

                        try:
                            sum_height = gridded_sum(grid_bounds, size, filtered_x, filtered_y, filtered_area_weighted_height)
                            sum_area = gridded_sum(grid_bounds, size, filtered_x, filtered_y, filtered_area)
                        except ValueError as e:
                            print(f"Skipping {wetland_poly}, grid size {size}: binned_statistic_2d failed with error: {e}")
                            continue
                        
                        weighted = np.full_like(sum_height, np.nan, dtype=float)
                        weighted_avg = np.divide(sum_height, sum_area, out=weighted, where=sum_area!=0) # Weighted average WSE
                        if not np.isnan(weighted_avg).all():
                            ave = np.nanmean(weighted_avg)
                            std = np.nanstd(weighted_avg)
                            outlier_removed = np.where((weighted_avg >= ave - 3 * std) & (weighted_avg <= ave + 3 * std), weighted_avg, np.nan)
                            gridded_height_anomaly = (outlier_removed - np.full_like(outlier_removed, np.nanmean(outlier_removed)))
                            counts, bins, patches = ax[1].hist(gridded_height_anomaly[~np.isnan(gridded_height_anomaly)], bins=np.arange(-0.3, 0.3, 0.01), edgecolor='black')

                            for center, patch in zip(bin_centers, patches):
                                color = cmap(norm(center))
                                patch.set_facecolor(color)

                            ax[1].set_ylabel('Grid frequency', fontsize=11)
                            fig.supxlabel('Water surface elevation anomaly (m)', fontsize=11)
                            plt.savefig(f'{save_folder}/pixel_grid_water_height_anomaly_histograms_{wetland_poly}_{str(date_obj)[0:10]}.png', bbox_inches='tight', pad_inches=0.2, dpi=600)
                            plt.show()

                            for variable, label, cmap in zip([outlier_removed, gridded_height_anomaly], ['Water surface elevation (m)', 'Water surface elevation anomaly (m)'], ['Blues', blue_white_green_cmap]):
                                fig, ax = plt.subplots()
                                ax.imshow(img, extent=img_extent) # Extent matched to actual coordinates
                                height, width = variable.shape
                                x_edges = np.linspace(min_x, max_x, width + 1)
                                y_edges = np.linspace(min_y, max_y, height + 1)
                                lon_x_edges, _ = transformer_reverse.transform(x_edges, np.full_like(x_edges, np.min(y_edges)))
                                _, lat_y_edges = transformer_reverse.transform(np.full_like(y_edges, np.min(x_edges)), y_edges)
                                mesh = ax.pcolormesh(lon_x_edges, lat_y_edges, np.flipud(variable), shading='auto', cmap=cmap)
                                cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
                                cbar.set_label(label=label, fontsize=11)
                                ax.ticklabel_format(useOffset=False)
                                plt.xlim(bounds[0]-0.001, bounds[1]+0.002)
                                plt.ylim(bounds[2]-0.001, bounds[3]+0.003)
                                plt.xlabel('Longitude', fontsize=11)
                                plt.ylabel('Latitude', fontsize=11)
                                plt.savefig(f'{save_folder}/spatial_{label}_{wetland_poly}_{str(date_obj)[0:10]}.png', bbox_inches='tight', pad_inches=0, dpi=600)
                                plt.show()     
    return None

#####################################
## Gridded Water Surface Elevation ##
#####################################

def compute_gridded_wetland_standard_deviation(directory, wetland_polygons, mask_function, save_to_filename):
    '''
    Computes the standard deviation of masked, filtered and gridded SWOT water surface elevations for grid cell sizes between 50 and 500 m inclusive at 50 m intervals for each wetland polygon over time. 
    Dataframes of the standard deviations for each grid size at each epoch are saved for each lake as .parquet files. Lakes with partial polygon coverage by the SWOT PIXC file are not processed. 

    Inputs: 
        directory: directory containing SWOT PIXC files
        wetland_polygons: dictionary of wetland polygons
        mask_function: function to use for masking out poor quality data
        save_to_filename: string to be included in the file name when saving
    
    Returns: 
        wetland_dataframes: list of wetland dataframes containing the standard deviations of the gridded wetland water surface elevation for different grid sizes at each epoch with full spatial coverage
    '''
    wetland_bounds = get_wetland_bounding_box(wetland_polygons)
    wetland_dataframes = [pd.DataFrame({'Grid Size': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]}) for _ in range(len(wetland_polygons))]

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        date = filename[29:44]
        date_format = '%Y%m%dT%H%M%S'
        date_obj = datetime.strptime(date, date_format)        
        ds1 = nc.Dataset(f)
        swot_polygon = Polygon([(getattr(ds1, 'inner_first_longitude'), getattr(ds1, 'inner_first_latitude')), (getattr(ds1, 'inner_last_longitude'), getattr(ds1, 'inner_last_latitude')), (getattr(ds1, 'outer_last_longitude'), getattr(ds1, 'outer_last_latitude')), (getattr(ds1, 'outer_first_longitude'), getattr(ds1, 'outer_first_latitude'))])
        data = read_pixc_values_tide(ds1) # Call function to read in data from SWOT PIXC file

        for wetland_poly, grid_std, bounds in zip(wetland_polygons, wetland_dataframes, wetland_bounds):
            if not swot_polygon.contains(wetland_polygons[wetland_poly]):
                # print(f'Insufficient wetland polygon coverage: skipping {wetland_poly}')
                continue
            study_site_mask = (data[1] > bounds[2]) & (data[1] < bounds[3]) & (data[2] > bounds[0]) & (data[2] < bounds[1])
            filtered_arrays = [array[study_site_mask] for array in data]
            [height, latitude, longitude, classification, sigma0, geoid, intf_qual, geolocation_qual, pixel_area, classification_qual, sig0_qual, solid_earth_tide, load_tide_fes, load_tide_got, pole_tide] = filtered_arrays

            mask = mask_function(classification, sigma0, intf_qual, geolocation_qual, classification_qual, sig0_qual) # Create mask to remove poor quality data
        
            # Apply mask and tidal models: Note that load_tide_fes is used, not load_tide_got
            Masked_Pixels = (height - geoid - solid_earth_tide - load_tide_fes - pole_tide) * mask
            Masked_Height = Masked_Pixels[~np.isnan(Masked_Pixels)]
            Masked_pixel_area = pixel_area[~np.isnan(Masked_Pixels)]
            latitude = latitude[~np.isnan(Masked_Pixels)]
            longitude = longitude[~np.isnan(Masked_Pixels)]

            if len(latitude) > 0:
                easting, northing = transformer.transform(longitude, latitude)
                polygon = transform(transformer.transform, wetland_polygons[wetland_poly])
                df = pd.DataFrame({'x': easting, 'y': northing, 'height': Masked_Height, 'area_weighted': Masked_Height*Masked_pixel_area, 'area':Masked_pixel_area})
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
                polygon_gdf = df[gdf.geometry.intersects(polygon)]
                polygon_df = polygon_gdf.copy()
                x,y = polygon.exterior.xy
                grid_bounds = (np.min(x), np.min(y), np.max(x), np.max(y)) 

                if len(polygon_df) > 50:
                    if len(polygon_df) > 1000:
                        kde = gaussian_kde(polygon_df['height'], bw_method=0.05)
                    elif (len(polygon_df) <= 1000) and (len(polygon_df) > 100):
                        kde = gaussian_kde(polygon_df['height'], bw_method=0.1)
                    elif (len(polygon_df) <= 100) and (len(polygon_df) > 50):
                        kde = gaussian_kde(polygon_df['height'], bw_method=0.2)

                    x = np.linspace(min(polygon_df['height']), max(polygon_df['height']), 1000)
                    kde_density = kde(x)
                    threshold1 = np.max(kde_density) * 0.2
                    left_tolerance1 = x[np.argmax(kde_density) - np.argmax((kde_density[:np.argmax(kde_density)] < threshold1)[::-1]) - 1]
                    right_tolerance1 = x[np.argmax(kde_density[np.argmax(kde_density):] < threshold1) + np.argmax(kde_density)]
                    filtered_df = polygon_df[(polygon_df['height'] >= left_tolerance1) & (polygon_df['height'] <= right_tolerance1)]
                else:
                    filtered_df = polygon_df

                standard_deviations = []
                if len(filtered_df) >= 5:
                    for size in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
                        filtered_x = filtered_df['x'].values
                        filtered_y = filtered_df['y'].values
                        filtered_area_weighted_height = filtered_df['area_weighted'].values
                        filtered_area = filtered_df['area'].values

                        try:
                            sum_height = gridded_sum(grid_bounds, size, filtered_x, filtered_y, filtered_area_weighted_height)
                            sum_area = gridded_sum(grid_bounds, size, filtered_x, filtered_y, filtered_area)
                        except ValueError as e:
                            print(f"Skipping {wetland_poly}, grid size {size}: gridded dataframe sum failed with error: {e}")
                            standard_deviations.append(np.nan)
                            continue
                        
                        weighted = np.full_like(sum_height, np.nan, dtype=float)
                        weighted_avg = np.divide(sum_height, sum_area, out=weighted, where=sum_area!=0) # Area-weighted average water surface elevation

                        if not np.isnan(weighted_avg).all():
                            ave = np.nanmean(weighted_avg)
                            std = np.nanstd(weighted_avg)
                            outlier_removed = np.where((weighted_avg >= ave - 3 * std) & (weighted_avg <= ave + 3 * std), weighted_avg, np.nan)
                            if len(outlier_removed[~np.isnan(outlier_removed)]) >= 5:
                                std1 = np.nanstd(outlier_removed)
                                standard_deviations.append(std1)
                            else:
                                standard_deviations.append(np.nan)
                    grid_std[str(date_obj)[0:10]] = standard_deviations    
    for wetland_poly, grid_std in zip(wetland_polygons, wetland_dataframes):
        grid_std.to_parquet(f'{save_folder}/std_{save_to_filename}_{wetland_poly}.parquet')
    return wetland_dataframes

def plot_grid_size_stdev(wetland_polygons, filename_string, poor_coverage_wetland_dictionary):
    '''
    Plots the average standard deviation for each wetland over time for each grid size. Dates identified to have insufficient wetland coverage can be removed. 
    The plot is saved to a .png file. 

    Inputs: 
        wetland_polygons: dictionary containing wetland polygons
        filename_string: save_to_filename defined in the compute_gridded_wetland_standard_deviation function
        poor_coverage_wetland_dictionary: dictionary containing a list of dates for each wetland in 'YYYY-MM-DD' format where there is insufficient wetland polygon coverage after masking and filtering 
    '''
    colours = colour_sequence(len(wetland_polygons), 'Dark2')
    plt.figure(figsize=(8,4)) 
    wetland_df = []

    for wetland_poly in wetland_polygons:
        df = pd.read_parquet(f'{save_folder}/std_{filename_string}_{wetland_poly}.parquet')
        wetland_df.append(df)

    for i, (wetland_name, df) in enumerate(zip(wetland_polygons, wetland_df)):
        data = df.drop(columns = 'Grid Size')
        for Date in poor_coverage_wetland_dictionary[wetland_name]:
            data = data.drop(columns=[Date])

        grid_list = []
        average_standard_dev = []
        for idx, size in enumerate([50, 100, 150, 200, 250, 300, 350, 400, 450, 500]):
            ave_std = np.nanmean(data.iloc[idx])
            average_standard_dev.append(ave_std)
            grid_list.append(size)
        plt.plot(grid_list, average_standard_dev, color=colours[i], label=wetland_name)

    plt.legend(loc='best')
    plt.xlabel('Grid size (m)')
    plt.ylabel('Standard deviation (m)')
    plt.savefig(f'{save_folder}/grid_size_stdev_mask_{filename_string}.png', dpi=900)
    plt.show()
    return None

