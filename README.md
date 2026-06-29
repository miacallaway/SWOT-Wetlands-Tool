# Surface Water and Ocean Topography satellite data reveals flood depths and environmental flows in wetlands

Mia Callaway, Maya Taib, Paul Tregoning, Chris Gouramanis and Lachlan Dodd

Contact emails: <Mia.Callaway@anu.edu.au>, <Maya.Taib@anu.edu.au>, <Paul.Tregoning@anu.edu.au>

Research School of Earth Sciences

The Australian National University

*Communications Earth & Environment*, accepted in principle June 2026

### Abstract
> Wetland ecosystems depend on water availability to function. However, detailed information on water surface elevations and depths within wetlands is often lacking, and management of these vulnerable ecosystems often rely on distant river gauge measurements or no water observations at all. Here we show that Surface Water and Ocean Topography (SWOT) satellite data can provide water surface elevation estimates across internationally recognised Ramsar wetlands in Australia - Coongie Lakes and the Barmah-Millewa Forest - with 3-10 cm accuracy. Combined with high-resolution elevation data, wetland water depths can now be monitored every ~10-11 days, providing crucial information on water availability to sustain ecosystems. The largest ever satellite-observed flood in the Coongie Lakes was tracked in detail as it flowed through interconnected channels and lakes of the system, and we mapped wetland extents from the SWOT data. Our approach can improve management of all wetlands lacking information on water surface elevations and depths.

### How is it done?

We process SWOT Pixel Cloud (PIXC) data to generate spatially gridded water depth estimates, wetland profiles, gridded water height anomalies and time series of wetland-averaged water heights. Details on methods used are described in the Methods section of Callaway *et al.* (2026).

![flowchart](fig_flowchart.png)

### Software

This software requires Python 3.12.11 and the packages identified in the provided requirements.txt file. We provide two python scripts and a Jupyter notebook to permit users to perform the computations:

`requirements.txt`: List of all python packages and versions required for the following python scripts and Jupyter notebook.

`swot_wetland_functions.py`: Script containing functions and files to compute, plot and save outputs generated in the paper.

`site_specific_variable.py`: Study site specific variables used in the functions in swot_wetland_functions.py which need to be replaced for the wetland of interest.

`paper_output_code.ipynb`: Script to call the functions in the swot_wetland_functions.py script and generate the main figure outputs presented in the paper.

`swot_wetlands.tar.gz`: Tar file containing all repository files.

### Input Data Files

All data used to produce the main figures in the paper can be downloaded from their native source locations. SWOT Level 2 Water Mask Pixel Cloud Data Product, Version C and D data are available from [NASA Earthdata](https://search.earthdata.nasa.gov/search?q=SWOT). The in situ gauge data used in this study are available from the [Australian Bureau of Meteorology](https://www.bom.gov.au/waterdata/). LiDAR data for Coongie Lakes is available from [ELVIS](https://elevation.fsdf.org.au/). Sentinel 2 data are available from the [Copernicus Browser](https://browser.dataspace.copernicus.eu/). The Digital Earth Australia Waterbodies dataset is available at <https://doi.org/10.26186/148920>.

### Citation

This work is described in detail in Callaway *et al.* (2026). Please cite this paper if you use this code.

M. Callaway, M. Taib, P. Tregoning, C. Gouramanis and L. Dodd (2026), "Surface Water and Ocean Topography satellite data reveals flood depths and environmental flows in wetlands", *Communications Earth & Environment*, accepted in principle June 2026.
