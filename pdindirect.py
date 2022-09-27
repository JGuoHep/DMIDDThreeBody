# using python to write likelihood part.

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import brute, fmin, minimize_scalar
from scipy.special import gammainc
import os

jfactor_path = "/home/jguo/workspace/data/Fermi_Data/"

dwarves_list = ['carina', 'draco', 'fornax','leo_I','leo_II','sculptor','sextans',
                 'ursa_minor', 'bootes_I', 'canes_venatici_II', 'coma_berenices', 
                 'hercules', 'segue_1', 'ursa_major_II', 'willman_1']
# dwarves_list = ['coma_berenices', 'draco']


class WhiteDWarfs():

    def __init__(self, dwares=dwarves_list, jfactor_path=jfactor_path):
        self.dwares = dwares
        self.jfactor_path = jfactor_path
        self.dw_infos = self.get_dw_infos

    def get_jfactors(self):
        dwarves_dict = {}
        jfactor_data = np.loadtxt(self.jfactor_path+"Jfactors.dat")
        with open(self.jfactor_path+"Jfactors.dat") as f:
            for line in f.readlines():
                if ("Row" in line):
                    _, _, num, dwarve = line.split()
                    dwarves_dict[dwarve] = int(num)
        dwarves_jfactor = np.array([jfactor_data[dwarves_dict[dw]] for dw in self.dwares])
        jfactor_roi = np.stack([dwarves_jfactor[:, 3], dwarves_jfactor[:, 4]])
        return jfactor_roi

    def get_ll_tabels(self):
        ll_tables = [np.loadtxt(self.jfactor_path+"likelihoods/like_"+dw+".txt") for dw in self.dwares]
        ll_tables = np.array(ll_tables)
        return ll_tables

    @property
    def get_dw_infos(self):
        dw_info_list = []
        jfactor_info = self.get_jfactors()
        ll_tables = self.get_ll_tabels()
        n_dw = ll_tables.shape[0]
        for i in range(n_dw):
            dw_i_dict = {}
            dw_i_dict['mu_j']  = jfactor_info[0][i]
            dw_i_dict['sigma_j']  = jfactor_info[1][i]
            dw_i_dict['ll_table'] = ll_tables[i]
            dw_info_list.append(dw_i_dict)
        self._dw_infos = dw_info_list  # np.array(dw_info_list)
        return self._dw_infos


class FermiPvalue():

    def __init__(self, whitedwarfs, spectrum_path, mdm):
        self.whitedwarfs = WhiteDWarfs().dw_infos # class of WhiteDwarfs
        self.nBin = 24
        self.j0 = 3.086e21
        self.sigmav0 = 1e-26
        self.mdm = mdm
        self.spectrum_path = spectrum_path
        self.spectrum = self.get_spectrum
        self.spectrum_path = spectrum_path

    @property
    def get_spectrum(self):
        fe2dndx = np.loadtxt(self.spectrum_path) # fe is some function of energy, dnde is dn/de.
        fe = fe2dndx[:, 0] # f(E) = np.log10(E/mdm).
        e = np.power(10, fe) * self.mdm
        dndx = fe2dndx[:, 1]
        dnde = dndx / (e * np.log(10.))
        self._spectrum = np.stack([e, dnde])
        return self._spectrum

    def  espectrum_e(self, e):
        '''Get the dn/de of certain E, return a function object calculating e * dn/de.'''
        energy = self.spectrum[0]
        dnde = self.spectrum[1]
        logfspectrum = interp1d(np.log10(energy), np.log10(dnde))
        if (e<energy.max() and e>energy.min()):
            return np.power(10, logfspectrum(np.log10(e)))* e  # np.exp(logfspectrum(np.log(e)))[0] * e
        else:
            return 0.0

    def get_up_dowm(self):
        """Get the up and down bound."""
        bin_down = self.whitedwarfs[0]["ll_table"][:, 0]  # get the energy bins from first dwarf likelihood table.
        bin_down = np.array(list(set(bin_down)))*1e-3
        bin_down = np.sort(bin_down)
        bin_up = self.whitedwarfs[0]["ll_table"][:, 1]  # get the energy bins from first dwarf likelihood table.
        bin_up = np.array(list(set(bin_up)))*1e-3
        bin_up = np.sort(bin_up)
        return bin_up, bin_down

    def get_predict_fluxes(self, sigmav):
        """Get the predict fluxes by the spectrum."""
        bin_down = self.whitedwarfs[0]["ll_table"][:, 0]  # get the energy bins from first dwarf likelihood table.
        bin_down = np.array(list(set(bin_down)))*1e-3
        bin_down = np.sort(bin_down)
        bin_up = self.whitedwarfs[0]["ll_table"][:, 1]  # get the energy bins from first dwarf likelihood table.
        bin_up = np.array(list(set(bin_up)))*1e-3
        bin_up = np.sort(bin_up)
        tol = min(self.espectrum_e(1e2),self.espectrum_e(1e5))*1e-10
        predict_fluxes = [quad(self.espectrum_e, down, up, epsabs=tol)[0] for down, up in zip(bin_down, bin_up)]
        predict_fluxes = np.array(predict_fluxes)
        predict_fluxes = predict_fluxes * self.sigmav0 * self.j0 / (4 * 2 * np.pi * self.mdm**2)
        predict_fluxes = predict_fluxes * (sigmav/self.sigmav0)
        return predict_fluxes

    def _dwraf_likelihood(self, dwarf, predict_fluxes):
        """Get the log likelihood of one dwarf."""
        j = dwarf['mu_j']
        sigma_j = dwarf['sigma_j']
        ll_table = dwarf["ll_table"]
        n_bin = len(set(ll_table[:, 0]))
        flux2ll = ll_table[:, 2:].reshape([n_bin, -1, 2])
        flux2ll_funcs = [interp1d(flux2ll[i][:, 0] *1e-3, flux2ll[i][:, 1], bounds_error=False, fill_value=-1e5) for i in range(n_bin)]
        def likeli(x):
            fluxes = predict_fluxes * 10**(j+x*sigma_j)/self.j0
            likeli_flux = np.array([func(flux) for func, flux in zip(flux2ll_funcs, fluxes)]).sum()
            likeli_jfactor = - 0.5*x**2 * np.log(np.sqrt(2*np.pi))
            likeli_dm = -1 * (likeli_flux + likeli_jfactor)  # Get minimum.
            return likeli_dm
        optimizer = minimize_scalar(likeli,method='bounded',bounds=(-5.0,5.0))
        likeli_dm = -optimizer.fun
        likeli_null = np.array([func(0) for func in flux2ll_funcs]).sum()
        return likeli_null, likeli_dm

    def dwarfs_likelihood(self,sigmav):
        """Get the combiine dwarfs loglikelihoods, including predict
        likelihood and null likelihood.
        """
        predict_fluxes = self.get_predict_fluxes(sigmav)
        dwarfs = self.whitedwarfs
        likelihoods = []
        for dwarf in dwarfs:
            likelihood = self._dwraf_likelihood(dwarf, predict_fluxes)
            likelihoods.append(likelihood)
        likelihoods = np.array(likelihoods)
        likelihoods = np.sum(likelihoods, axis=0)
        ts = 2 * (likelihoods[0] - likelihoods[1])
        pval = 1-gammainc(1/2.0, ts/2.0)
        return 1 - pval


class GetExcluding():

    def __init__(self, sigmavrange, spectrum_path, cls):
        self.sigmavrange=sigmavrange
        self.cls = cls
        self.whitedwarfs = WhiteDWarfs().dw_infos
        self.spectrum_path = spectrum_path

    def loss(self, fermip, sigmav):
        pval = fermip.dwarfs_likelihood(sigmav)
        atanhpval = np.arctanh(pval - 1e-9)
        atanclval = np.arctanh(self.cls - 1e-9)
        loss_value = (atanhpval - atanclval)**2
        return loss_value

    def get_excluding_point(self, mdm):
        """Get the up limit of sigmav with DM mass mdm."""
        fermip = FermiPvalue(self.whitedwarfs, self.spectrum_path, mdm)
        def loss_fn(logsigmav):
            sigmav = np.power(10., logsigmav)
            return self.loss(fermip, sigmav)
        logsigmav_min = np.log10(self.sigmavrange)[0]
        logsigmav_max = np.log10(self.sigmavrange)[1]
        n_steps = int(5 * (logsigmav_max - logsigmav_min))
        search = brute(loss_fn, [(logsigmav_min,logsigmav_max)], Ns=n_steps, full_output=False, finish=fmin)
        print(search[0])
        sigmav_ul = np.power(10, search[0])
        print("cls predict: ", fermip.dwarfs_likelihood(sigmav_ul))
        print("cls predict 2: ", fermip.get_predict_fluxes(0))
        return sigmav_ul

    def get_excluding_line(self, mrange):
        """Get the excluding line in DM mass range mrange in `GeV`"""
        mdm_down = mrange[0]
        mdm_up = mrange[1]
        mass = []
        sigmav_uls = []
        for mdm in range(mdm_down, mdm_up):
            sigmav_ul = self.get_excluding_point(mdm)
            print("one step!", sigmav_ul)
            mass.append(mdm)
            sigmav_uls.append(sigmav_ul)
        masses = np.array(mass)
        sigmav_uls = np.array(sigmav_uls)
        excluding_line = np.concatenate([masses, sigmav_uls], axis=1)
        return excluding_line


if __name__ == '__main__':
    cls = 0.9
    mmin = 130
    mmax = 10000
    spectrum_path = "./gammas_spectrum.dat"
    step = (np.log10(mmax) - np.log10(mmin))/20.
    mrange = np.arange(np.log10(mmin), np.log10(mmax), step)
    print(mrange)
    sigmavrange = np.array([1e-28, 1e-24])
    result_f = open('result.txt', 'a')
    # excluding_line = excluding.get_excluding_line(mrange)
    for logmdm in mrange:
        mdm = np.power(10, logmdm)
        print("mdm: ", mdm)
        command = "./spectrummc {}".format(mdm)
        os.system(command)
        excluding = GetExcluding(sigmavrange, spectrum_path, cls)
        sigmav_ul = excluding.get_excluding_point(mdm)
        result_f.write("{}     {}\r".format(mdm, sigmav_ul))
    result_f.close()






