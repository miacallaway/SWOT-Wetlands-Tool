'''
Site-specific variables used by the functions in 'SWOT_wetlands_functions.py'. Examples are shown for Coongie Lakes and Barmah-Millewa Forest.
The functions used to generate the outputs shown in our paper can also be applied to other wetland sites globally by substituting the parameters defined in the 
'site_specific_variables.py' file. Examples inputs for the wetlands we investigated are shown. 

This file was used during the production of the paper 'SWOT reveals flood depths and environmental flows in wetlands' - Callaway et al., submitted October 2025
'''
#############################################
## Coongie Lakes and Barmah-Millewa Forest ##
#############################################
save_folder = 'Paper Output Data'                        # Folder to save all output data

###########################
## Coongie Lakes Example ##
###########################
directory = 'Coongie Lakes/'                             # Directory containing all SWOT PIXC netCDF files for the study site
epsg_xy_m = 'EPSG:28354'                                 # EPSG code for study site in x,y (m units): e.g. EPSG:28354 (GDA94, MGA Zone 54) is the projection for Coongie Lakes
image_file = 'Coongie_Lakes_satellite.png'               # path to satellite image of study area
img_extent = [140.0, 140.5, -27.4, -26.9]                # extent of satellite image in [minimum_longitude, maximum_longitude, minimum_latitude, maximum_latitude]
xmin, xmax, ymin, ymax = 140.11, 140.425, -27.24, -26.92 # study site extent as [minimum_longitude, maximum_longitude, minimum_latitude, maximum_latitude]

# Tranformation pipeline to convert SWOT WGS84 to GDA94 + AHD height
wgs84_to_gda94_ahd_pipeline = """proj=pipeline step proj=axisswap order=2,1 step proj=unitconvert xy_in=deg xy_out=rad step inv proj=vgridshift grids=au_ga_AUSGeoid09_V1.01.tif multiplier=1 step proj=unitconvert xy_in=rad xy_out=deg step proj=axisswap order=2,1"""

##################################
# Barmah-Millewa Forest Example ##
##################################
# directory = 'BM SWOT Data/'
# epsg_xy_m = 'EPSG:28355' # GDA94, MGA Zone 55 (for Barmah-Millewa Forest)
# image_file = 'BM_satellite_basemap.png'
# img_extent = [144.8, 145.5, -36.2, -35.5]
# xmin, ymin, xmax, ymax = 144.908, -35.965, 145.095, -35.825