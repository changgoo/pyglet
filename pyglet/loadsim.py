from . import athena_read as ar
import numpy as np
import xarray as xr
from pathlib import Path
import yt

def _parse_athdf_filename(fname):
    sp = fname.split('.')
    outid, num, suffix = sp[-3:]
    problem_id = '.'.join(sp[:-3])

    return problem_id, outid, num, suffix

class LoadSim(object):
    """Class for preparing Athena++ simulation data analysis.

    Parameters
    ----------
    basedir : str
            Name of the directory where all data is stored

    Methods
    -------
    load_athdf : Reads hdf5 (.athdf) output file
    load_hst : Reads history dump (.hst)

    Attributes
    ------------
    basedir : str
        base directory of simulation output
    basename : str
        basename (tail) of basedir
    files : dict
        output file paths for athdf, hst, rst
    problem_id : str
        prefix for (athdf, hst, rst) output
    meta : dict
        simulation metadata (information in athinput file)
    nums : list of int
        athdf output numbers
    """

    def __init__(self, basedir):
        """Constructor for LoadSim class.
        """

        # Use pathlib.Path for handling file paths
        self.basedir = Path(basedir)
        self.basename = self.basedir.name

        # Find output files by matching glob patterns
        self.files = {}
        patterns = dict(athinput='athinput.*',
                        hst='*.hst',
                        athdf='*.athdf',
                        rst='*.rst',
                        partab='*par?.tab',
                        parhst='*par?.csv',
                        stdout='slurm-*.out',
                        stderr='slurm-*.err',
                        coolftn='*coolftn.txt') # add additional patterns here
        for key, pattern in patterns.items():
            self.files[key] = sorted(self.basedir.glob(pattern))
            if len(self.files[key]) == 0:
                print("WARNING: Found no {} file".format(key))
        if len(self.files['hst']) > 1:
            print("WARNING: Found more than one history files")
        if len(self.files['athinput']) > 1:
            print("WARNING: Found more than one input files")

        # Get metadata from standard output file
        try:
            with open(self.files['stdout'][-1], 'r') as stdout:
                print("Reading metadata from the last stdout file: {}".format(self.files['stdout'][-1].name))
                niter = 0
                toggle = False
                lines = []
                for line in stdout:
                    if niter > 1000:
                        print("Cannot find PAR_DUMP block in the first 1000 lines")
                        break
                    if toggle:
                        lines.append(line.split('#')[0].strip())
                    if 'PAR_DUMP' in line:
                        toggle = not toggle
                    niter += 1
                # remove empty lines
                lines = filter(None, lines)
            self.meta = ar.athinput(None, lines)
            self.problem_id = self.meta['job']['problem_id']
        except IndexError:
            print("WARNING: Failed to read metadata from the standard output file.")
            if len(self.files['athdf'])>0:
                problem_id, _, _, _, = _parse_athdf_filename(self.files['athdf'][0].name)
                self.problem_id = problem_id
        if len(self.files['coolftn'])>0:
            if self.files['coolftn'][0].name == 'tigress_coolftn.txt':
                self.tigress_cooling = True
                import pandas as pd
                from scipy.interpolate import interp1d
                from .units import Units
                self.u = Units(kind='LV', muH=1.4271)
                cf=pd.read_csv(self.files['coolftn'][0]).rename(columns={'#rho':'rho'})
                cf['T1'] = cf['Press']*self.u.pok/(self.u.muH*cf['rho'])
                self.get_temp = interp1d(cf['T1'],cf['Temp'],fill_value="extrapolate")
                self.get_Lambda = interp1d(cf['T1'],cf['cool'],fill_value="extrapolate")
                self.get_Gamma = interp1d(cf['T1'],cf['heat'],fill_value="extrapolate")
        # Find athdf output numbers
        self.nums=dict()
        for f in self.files['athdf']:
            _, outid, num, _, = _parse_athdf_filename(f.name)
            if outid in self.nums:
                self.nums[outid].append(int(num))
            else:
                self.nums[outid] = [int(num)]
        for k in self.nums:
            self.nums[k] = sorted(self.nums[k])
        # self.nums = sorted(map(lambda x: int(x.name.removesuffix('.athdf')[-5:]),
        #                        self.files['athdf']))


    def load_athdf(self, num=None, output_id=None, load_method='xarray'):
        """Read Athena hdf5 file and convert it to xarray Dataset

        Parameters
        ----------
        num : int
            Snapshot number, e.g., /basedir/problem_id.00042.athdf
        output_id : int (optional)
            Output id given in the <output#> block. Useful when there are
            more than one athdf outputs.
        load_method : str (optional)
            Method to read athdf file. Available options are 'xarray' or 'yt'.
            Default is 'xarray'

        Returns
        -------
        dat : xarray.Dataset
        """

        if output_id is None:
            # Find output_id of hdf5 files
            fname = self.files['athdf'][0].name
            idx = fname.find('.out')
            output_id = fname[idx+4]

        if load_method=='xarray':
            # Read athdf file using athena_read
            dat = ar.athdf(self.basedir / '{}.out{}.{:05d}.athdf'.format(
                           self.problem_id, output_id, num))

            # Convert to xarray object
            varnames = set(map(lambda x: x.decode('ASCII'), dat['VariableNames']))
            variables = [(['z', 'y', 'x'], dat[varname]) for varname in varnames]
            attr_keys = (set(dat.keys()) - varnames
                         - {'VariableNames','x1f','x2f','x3f','x1v','x2v','x3v'})
            attrs = {attr_key:dat[attr_key] for attr_key in attr_keys}
            for xr_key, ar_key in zip(['dx','dy','dz'], ['x1f','x2f','x3f']):
                dx = np.unique(np.diff(dat[ar_key])).squeeze()
                if dx.size == 1: dx = dx[()]
                attrs[xr_key] = dx
            if hasattr(self,'meta'): attrs['meta'] = self.meta
            ds = xr.Dataset(
                data_vars=dict(zip(varnames, variables)),
                coords=dict(x=dat['x1v'], y=dat['x2v'], z=dat['x3v']),
                attrs=attrs
            )
        elif load_method=='yt':
            ds = self.ytload(self.basedir / '{}.out{}.{:05d}.athdf'.format(
                             self.problem_id, output_id, num))
        return ds

    def load_hst(self):
        """Reads athena++ history dump and convert it to xarray Dataset

        Returns
        -------
        hst : xarray.Dataset
              coordinate
                  - t: simulation time
              variables
                  - dt: simulation timestep
                  - mass: total mass
                  - mom1, mom2, mom3: total momenta in x, y, z-dir.
                  - KE1, KE2, KE3: total kinetic E in x, y, z-dir.
                  - gravE: total self-grav. potential E (\int 0.5*rho*Phi dV)
        """
        hst = ar.hst(self.files['hst'][0])
        data_vars = {key: ('t', hst[key]) for key in hst.keys()}
        coords = dict(t=hst['time'])
        hst = xr.Dataset(data_vars, coords).drop('time')
        hst = hst.rename({f'{i}-mom':f'mom{i}' for i in [1,2,3]}
                         | {f'{i}-KE':f'KE{i}' for i in [1,2,3]})
        if 'grav-E' in hst:
            hst = hst.rename({'grav-E':'gravE'})

        return hst

    def ytload(self,fname):
        self.u.units_override.update(dict(magnetic_unit=(self.u.muG*1.e-6,"gauss")))

        # define fields from TIGRESS-NCR output
        from yt.utilities.physical_constants import mh, me, kboltz

        muH = self.u.muH

        def _ndensity(field, data):
            return data[("gas","density")]/(muH*mh)

        def _nelectron(field, data):
            hot = data[("gas", "temperature")] > 3.5e4
            return data[("gas","density")]*hot*1.2/(muH*mh)

        def _T1(field, data):
            return data[("gas","pressure")]/data[("gas","H_nuclei_density")]/(muH*kboltz)

        def _temperature(field,data):
            return self.get_temp(data[("gas","T1")])*yt.units.K

        def _EM(field, data):
            return data[("gas","H_nuclei_density")]*data[("gas","El_number_density")]*data[("gas","cell_volume")]

        # def _metallicity(field, data):
        #     return data[("athena","specific_scalar[0]")]

        # def _metallicity_solar(field, data):
        #     return data[("athena","specific_scalar[0]")]/Zsolar

        ds = yt.load(fname,units_override=self.u.units_override,hint='AthenaPPDataset')
        # print(ds.field_list)
        # add/override fields
        ds.add_field(("gas","H_nuclei_density"),function=_ndensity, force_override=True,
                     units='cm**(-3)',display_name=r'$n_{\rm H}$', sampling_type="cell")
        ds.add_field(("gas","El_number_density"), function=_nelectron, force_override=True,
                     units='cm**(-3)',display_name=r'$n_{\rm e}$', sampling_type="cell")
        ds.add_field(("gas","T1"), function=_T1, force_override=True,
                     units='K',display_name=r'$T_\mu$', sampling_type="cell")
        ds.add_field(("gas","temperature"), function=_temperature, force_override=True,
                     units='K',display_name=r'T', sampling_type="cell")
        ds.add_field(("gas","emission_measure"), function=_EM, force_override=True,
                     units='cm**(-3)',display_name=r'EM', sampling_type="cell")
        # ds.add_field(("gas","metallicity"), function=_metallicity, force_override=True,
        #              units='dimensionless',display_name=r'Z', sampling_type="cell")
        # ds.add_field(("gas","metallicity_solar"), function=_metallicity_solar,
        #              force_override=True,
        #              units='dimensionless',display_name=r'$Z/Z_\odot$', sampling_type="cell")

        def ionized_gas(pfilter, data):
            pfilter1 = data[pfilter.filtered_type, "temperature"] > 3.5e4
            return pfilter1

        yt.add_particle_filter(
            "ionized_gas",
            function=ionized_gas,
            filtered_type="gas",
            requires=["temperature", "density"],
        )

        return ds