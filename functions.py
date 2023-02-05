import pandas as pd
import numpy as np
import xarray as xr
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy 
import cartopy.crs as ccrs


class aerosol_dehm:

    def __init__(self):
        self.infile = 'data/DEHM_output.nc'
        self.initial_load()
        self.par = 'SO4_ugSm-3'
        self.vname = 'Sulfate'
        self.unit = '$\u03BCg/m^3$'



    def initial_load(self):
        self.ds = xr.open_dataset(self.infile)


    def plot_annual_map(self,par=None,vmin=None,vmax=None):
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-32)})
        # arr = self.ds[self.par].mean(dim='time').values
        par = self.par if par is None else par
        arr = self.ds[par].values
        im = self.get_map(ax,arr,vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(im,extend='max')
        cbar.set_label(label=f'{self.vname} concentrations ({self.unit})')

        return fig

    def get_map(self,ax,arr,vmin=None,vmax=None):
        cmap = 'turbo'

        ax.coastlines(color='k',zorder=8,alpha=0.5,linewidth=0.3)
        gl = ax.gridlines(draw_labels=True,zorder=6,alpha=0.5,linewidth=0.3,\
            crs=ccrs.PlateCarree())
        gl.xlabel_style = {'size':4}
        gl.ylabel_style = {'size':4}

        cmap = plt.get_cmap(cmap)
        cmap.set_bad ('lightgrey',1.0)
        
        lat,lon = self.ds['lat'].values, self.ds['lon'].values

        # arr1 = np.where(arr<vmin,np.nan,arr)
        if vmin is None or vmax is None:
            im = ax.pcolormesh(lon, lat, arr, cmap=cmap,
                # vmin=vmin,vmax=vmax,
                transform=ccrs.PlateCarree(),
                zorder=0)
        else:
            im = ax.pcolormesh(lon, lat, arr, cmap=cmap,
                vmin=vmin,vmax=vmax,
                transform=ccrs.PlateCarree(),
                zorder=0)

        return im