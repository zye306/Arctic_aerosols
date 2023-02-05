import pandas as pd
import numpy as np
import xarray as xr
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy 
import cartopy.crs as ccrs


def calc_metrics(sim,obs):
    outdict = {}

    N = obs.shape[0]
    outdict['N'] = N

    meansim = np.nanmean(sim)
    meanobs = np.nanmean(obs)
    sigmasim = np.nanstd(sim)
    sigmaobs = np.nanstd(obs)

    outdict['mean_DEHM'] = meansim
    outdict['mean_OBS'] = meanobs
    outdict['std_DEHM'] = sigmasim
    outdict['std_OBS'] = sigmaobs

    diff = sim - obs
    MB = np.nanmean(diff)
    outdict['MB'] = MB

    square_diff = np.square(diff)
    mean_square_diff = np.nanmean(square_diff)
    RMSE = np.sqrt(mean_square_diff)
    outdict['RMSE'] = RMSE

    addition = np.absolute(sim)+np.absolute(obs)
    division = np.where(addition==0, np.nan, np.true_divide(diff, addition))
    NMB = 2*np.nanmean(division)
    outdict['NMB'] = NMB
    
    diffsim = sim - meansim
    diffobs = obs - meanobs
    multidiff = np.multiply(diffsim, diffobs)
    CORR = np.nanmean(multidiff)/(sigmasim*sigmaobs)
    # print('For CORR calculations: ')
    outdict['COV'] = np.nanmean(multidiff)
    outdict['corr'] = CORR

    return outdict



class aerosol_dehm:
    par_dict = {
                'SO2':{'par':'SO2_ugSm-3','name':'SO2','units':'$\u03BCgS/m^3$','vmin':-0.2,'vmax':3},\
                'SO4':{'par':'SO4_ugSm-3','name':'Sulfate','units':'$\u03BCgS/m^3$','vmin':-0.2,'vmax':1},\
                'tSO4':{'par':'SO4_ugSm-3','name':'Total sulfate','units':'$\u03BCgS/m^3$','vmin':-0.2,'vmax':1},\
                'NH4':{'par':'NH4_ugNm-3','name':'Ammonium','units':'$\u03BCgN/m^3$','vmin':-0.2,'vmax':1.7},\
                'PM2.5':{'par':'PM2.5_ugm-3','name':'PM2.5','units':'$\u03BCg/m^3$','vmin':-0.2,'vmax':9},\
                'PM10':{'par':'PM10_ugm-3','name':'PM10','units':'$\u03BCg/m^3$','vmin':-0.2,'vmax':22},\
                'SIA':{'par':'SIA_ugm-3','name':'Secondary inorganic aerosols','units':'$\u03BCg/m^3$','vmin':-0.2,'vmax':6}
                }

    def __init__(self):
        self.infile = 'data/DEHM_output.nc'
        self.initial_load()


    def initial_load(self):
        self.ds = xr.open_dataset(self.infile)

    def map_to_dataframe(self,var):
        par = self.par_dict[var]['par']
        lat,lon = self.ds['lat'].values, self.ds['lon'].values
        outdict={'x':[],'y':[],'lat':[],'lon':[],par:[]}
        arr = self.ds[par].values
        for i in range(300):
            for j in range(300):
                outdict['x'].append(i)
                outdict['y'].append(j)
                outdict['lat'].append(lat[i,j])
                outdict['lon'].append(lon[i,j])
                outdict[var].append(arr[i,j])
        
        # print(outdict)
        outdf = pd.DataFrame.from_dict(outdict)
        # print(outdf)
        return outdf


    def plot_annual_map(self,var,vmin=None,vmax=None):
        par,vname,units = self.par_dict[var]['par'],self.par_dict[var]['name'],self.par_dict[var]['units']
        vmin = self.par_dict[var]['vmin'] if vmin is None else vmin
        vmax = self.par_dict[var]['vmax'] if vmax is None else vmax

        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-32)})
        # arr = self.ds[self.par].mean(dim='time').values
        par = self.par if par is None else par
        arr = self.ds[par].values
        im = self.get_map(ax,arr,vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(im,extend='max')
        cbar.set_label(label=f'{vname} concentrations ({units})')
        cbar.ax.set_ylim([max(0,vmin),vmax])

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