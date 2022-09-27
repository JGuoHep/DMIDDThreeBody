/*This file is used for generate DM DM -> h h final state gamma ray spectrum.*/

#include"../include/micromegas.h"
#include"../include/micromegas_aux.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<cmath>
#include <fstream>
#include <iostream>
#include<chrono> 
#include<random> 
#include<vector>
#include<iterator>
#include <algorithm>
#include <map>
#include <string>
using namespace std;

int spectrum(double mdm)
{
    /*Generate higgs final state*/
    int pdg_higgs = 25;
    sortOddParticles(NULL);

    /*array to store gamma ray spectrum.*/
    double SpA[NZ];
    /*Get the basic gamma ray spectrum*/
    basicSpectra(mdm,pdg_higgs,0, SpA);

    /*Calculate the dNdE we need, same as Pythia*/
    double eminh = -9.;
    double emaxh = 0.;
    double nbins = 250;
    double delta_bin = (emaxh-eminh)/nbins; 
    // open file.
    const char *outfile_name = (char*)"./gammas_spectrum.dat";
    ofstream outfile;
	outfile.open(outfile_name, ios::out | ios::trunc);
    cout<<SpA[0] <<endl;
    for(int i=2; i<nbins; ++i)
    {   
        /*float x = eminh + delta_bin * i;
        float energy = pow(10, x) * mdm;
        double dnde = SpectdNdE(energy, SpA);
        outfile<< x<< "    " << dnde*energy*log(10) << endl;*/

        float energy = exp(log(1.E-7)*pow((double)(i-1)/(double)(nbins),1.5)) * mdm;
        float x = log10(energy / mdm);
        float dndx = SpA[i] * log(10.);
        outfile<< x<< "    " << dndx << endl;
    }
}


int main(int argc,char** argv)
{
    double mdm = stod(argv[1]);// reinterpret_cast<double &>(argv[1]);
    spectrum(mdm);
    cout<< "done!"<<endl;
    return 1;
}