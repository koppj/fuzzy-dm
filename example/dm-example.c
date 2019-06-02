// ----------------------------------------------------------------------------
// Fuzzy Dark Matter Example
// ----------------------------------------------------------------------------
// Simulates neutrino interactions with Fuzzy DM
// ----------------------------------------------------------------------------
// Authors: Joachim Kopp (jkopp@fnal.gov)
//          Xiaoping Wang (xia.wang@anl.gov)
// ----------------------------------------------------------------------------
// GLoBES -- General LOng Baseline Experiment Simulator
// (C) 2002 - 2010,  The GLoBES Team
//
// GLoBES as well as this add-on are mainly intended for academic purposes.
// Proper credit must be given if you use GLoBES or parts of it. Please
// read the section 'Credit' in the README file.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// ----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <globes/globes.h>
#include "fuzzy-dm.h"

// min/max functions avoiding multiple evaluations of arguments
double fmin(double x, double y) { return x < y ? x : y; }
double fmax(double x, double y) { return x > y ? x : y; }

// Unit conversion (everything to eV or eV^-1)
#define GEV        1.0e9             // [eV/GeV]
#define CM         (5.076e4)         // [eV^-1 / cm]

int main(int argc, char *argv[])
{
  /* if no command line arguments are given, print usage message */
  if (argc < 2  ||  (strcasecmp(argv[1],"CPV")!=0  && strcasecmp(argv[1],"DM_CHI2")!=0) )
  {
    fprintf(stderr, "Usage:  ./dm-example [CPV|DM_CHI2]\n");
    fprintf(stderr, "  where  CPV     = CP sensitivity as function of true delta_CP\n");
    fprintf(stderr, "         DM_CHI2 = chi^2 as function of DM-neutrino coupling\n");
    return -1;
  }

  /* Initialize libglobes */
  setenv("GLB_PATH", "dune/DUNE_GLoBES_Configs", 1);
  glbInit(argv[0]);

  /* Select Fuzzy DM oscillation engine */
  /* Note: initializing and registering the oscillation engine fixes the number of
   * oscillation parameters, so this must be done at the very beginning of the
   * code, and definitely before any call to glbAllocParams() */
  dm_init_probability_engine(DM_SCALAR);
  glbRegisterProbabilityEngine(dm_get_n_osc_params(), &dm_probability_matrix,
    &dm_set_oscillation_parameters, &dm_get_oscillation_parameters, NULL);
  for (int i=0; i < dm_get_n_osc_params(); i++)
    glbSetParamName(dm_get_param_name(i), i);

  /* Initialize experiment NFstandard.glb */
  glbInitExperiment("DUNE_GLoBES.glb", &glb_experiment_list[0], &glb_num_of_exps); 
 
  /* Define standard oscillation parameters (NuFit 4.0) */
  double theta12 = asin(sqrt(0.318));
  double theta13 = asin(sqrt(0.02241));
  double theta23 = asin(sqrt(0.580));
  double delta   = 215. * M_PI/180.;
  double dm21    = 7.39e-5;
  double dm31    = 2.525e-3;

  double rho_dm  = 0.3 * GEV/(CM*CM*CM);
  double m_dm    = 1e-22;
  double y_dm    = 1e-23;
  double chi_dm  = y_dm * sqrt(2.*rho_dm) / m_dm;
  
  /* Initialize parameter vector(s) */
  glb_params true_values  = glbAllocParams();
  glb_params test_values  = glbAllocParams();
  glb_params input_errors = glbAllocParams();
  glb_projection proj     = glbAllocProjection();

  for (int i=0; i < glbGetNumOfOscParams(); i++)
  {
    glbSetOscParams(true_values,  0.0, i);
    glbSetOscParams(input_errors, 0.0, i);
    glbSetProjectionFlag(proj, GLB_FIXED, i);
  }
  glbDefineParams(true_values, theta12, theta13, theta23, delta, dm21, dm31);
  glbSetOscParamByName(true_values, 0., "M1");
  glbSetDensityParams(true_values, 1.0, GLB_ALL);
  glbSetDensityProjectionFlag(proj, GLB_FIXED, GLB_ALL);
  glbCopyParams(true_values, test_values);

  // FIXME FIXME FIXME
//      dm_set_dm_type(DM_SCALAR);
//      glbDefineParams(true_values, theta12, theta13, theta23, M_PI/3., dm21, dm31);
//      glbSetOscillationParameters(true_values);
//      printf("%g %g\n", glbVacuumProbability(2, 1, +1, 1.2, 1300.),
//                        glbVacuumProbability(2, 1, -1, 1.2, 1300.));//FIXME
//      dm_set_dm_type(DM_VECTOR_POLARIZED);
//      printf("%g %g\n", glbVacuumProbability(2, 1, +1, 1.2, 1300.),
//                        glbVacuumProbability(2, 1, -1, 1.2, 1300.));//FIXME
////      printf("%g %g\n", glbConstantDensityProbability(2, 1, +1, 1.2, 1300., 5.),
////                        glbConstantDensityProbability(2, 1, -1, 1.2, 1300., 5.));//FIXME
////      dm_set_dm_type(DM_VECTOR_POLARIZED);
////      printf("%g %g\n", glbConstantDensityProbability(2, 1, +1, 1.2, 1300., 5.),
////                        glbConstantDensityProbability(2, 1, -1, 1.2, 1300., 5.));//FIXME
//      getchar();

  /* CPV sensitivity as a function of true delta_CP */
  /* ---------------------------------------------- */
  if (strcasecmp(argv[1], "CPV") == 0)
  {
    /* write header of output file */
    printf("#    delta_CP        chi2(no CPV)    chi2(no CPV)    chi2(no CPV)  chi2(no CPV)\n");
    printf("#     (true)            no DM          scalar DM      vector DM     vector DM\n");
    printf("#                                                    (polarized)   (unpolarized)\n");
    printf("#\n");

    /* main loop over true delta_CP values */
    for (double delta_cp=0.0; delta_cp <= 2.*M_PI+1e-10; delta_cp += M_PI/100.0)
    {
      double chi2_noDM, chi2_DM_scalar, chi2_DM_vector_pol, chi2_DM_vector_unpol;

      /* set true delta_CP */
      glbSetOscParamByName(true_values, delta_cp, "DELTA_0");
     
      /* test no-CPV hypothesis in the absence of a neutrino--DM coupling */
      dm_set_dm_type(DM_SCALAR);
      glbSetOscParamByName(true_values, 0., "ABS_CHI_MUTAU");
      glbSetOscParamByName(true_values, 0., "ABS_CHI_TAUMU");
      glbSetOscParamByName(true_values, 0., "ABS_XI_3");
      glbSetOscillationParameters(true_values);
      glbSetRates();
      glbSetOscParamByName(test_values, 0.0,  "DELTA_0");
      chi2_noDM = glbChiSys(test_values,GLB_ALL,GLB_ALL);
      glbSetOscParamByName(test_values, M_PI, "DELTA_0");
      chi2_noDM = fmin(chi2_noDM, glbChiSys(test_values,GLB_ALL,GLB_ALL));

      /* now the same including neutrino couplings to scalar DM. */
      dm_set_dm_type(DM_SCALAR);
      glbSetOscParamByName(true_values, chi_dm, "ABS_CHI_MUTAU");
      glbSetOscParamByName(true_values, chi_dm, "ABS_CHI_TAUMU");
      glbSetOscParamByName(true_values, 0., "ABS_XI_3");
      glbSetOscillationParameters(true_values);
      glbSetRates();
      glbSetOscParamByName(test_values, 0.0,  "DELTA_0");
      chi2_DM_scalar = glbChiSys(test_values,GLB_ALL,GLB_ALL);
      glbSetOscParamByName(test_values, M_PI, "DELTA_0");
      chi2_DM_scalar = fmin(chi2_DM_scalar, glbChiSys(test_values,GLB_ALL,GLB_ALL));

      /* polarized vector DM */
      dm_set_dm_type(DM_VECTOR_POLARIZED);
      glbSetOscParamByName(true_values, chi_dm, "ABS_CHI_MUTAU");
      glbSetOscParamByName(true_values, chi_dm, "ABS_CHI_TAUMU");
      glbSetOscParamByName(true_values, 1., "ABS_XI_3");
      glbSetOscillationParameters(true_values);
      glbSetRates();
      glbSetOscParamByName(test_values, 0.0,  "DELTA_0");
      chi2_DM_vector_pol = glbChiSys(test_values,GLB_ALL,GLB_ALL);
      glbSetOscParamByName(test_values, M_PI, "DELTA_0");
      chi2_DM_vector_pol = fmin(chi2_DM_vector_pol, glbChiSys(test_values,GLB_ALL,GLB_ALL));

      /* unpolarized vector DM */
      dm_set_dm_type(DM_VECTOR_UNPOLARIZED);
      glbSetOscillationParameters(true_values);
      glbSetRates();
      glbSetOscParamByName(test_values, 0.0,  "DELTA_0");
      chi2_DM_vector_unpol = glbChiSys(test_values,GLB_ALL,GLB_ALL);
      glbSetOscParamByName(test_values, M_PI, "DELTA_0");
      chi2_DM_vector_unpol = fmin(chi2_DM_vector_unpol, glbChiSys(test_values,GLB_ALL,GLB_ALL));

      /* print output */
      printf("%15.10g  %15.10g %15.10g %15.10g %15.10g\n",
             delta_cp, chi2_noDM, chi2_DM_scalar, chi2_DM_vector_pol, chi2_DM_vector_unpol);
    }
  }


  /* chi^2 as a function of the DM--neutrino coupling */
  /* ------------------------------------------------ */
  if (strcasecmp(argv[1], "DM_CHI2") == 0)
  {
    /* Simulate SM event rates */
    glbSetOscParamByName(true_values, 0., "ABS_CHI_MUTAU");
    glbSetOscParamByName(true_values, 0., "ABS_CHI_TAUMU");
    glbSetOscillationParameters(true_values);
    glbSetRates();

    /* write header of output file */
    printf("#   DM coupling      chi2(no CPV)    chi2(no CPV)  chi2(no CPV)\n");
    printf("#     (true)           scalar DM      vector DM     vector DM\n");
    printf("#                                    (polarized)   (unpolarized)\n");
    printf("#\n");

    /* set up projection */
    glbDefineProjection(proj, GLB_FIXED, GLB_FREE, GLB_FIXED, GLB_FREE, GLB_FIXED, GLB_FIXED);
    glbSetDensityProjectionFlag(proj, GLB_FIXED, GLB_ALL);
    glbSetProjection(proj);
    glbSetCentralValues(true_values);
    glbSetInputErrors(input_errors);

    /* loop over DM--neutrino coupling */
    for (double log_y=-35.; log_y <= -21.+1e-10; log_y+=0.1)
    {
      double y_dm = exp(M_LN10 * log_y);
      double chi_dm  = y_dm * sqrt(2.*rho_dm) / m_dm;
      double chi2_scalar, chi2_vector_pol, chi2_vector_unpol;

      /* chi^2 for scalar DM fit */
      dm_set_dm_type(DM_SCALAR);
      glbSetOscParamByName(test_values, chi_dm, "ABS_CHI_MUTAU");
      glbSetOscParamByName(test_values, chi_dm, "ABS_CHI_TAUMU");
      glbSetOscParamByName(test_values, 0.,     "ABS_XI_3");
      chi2_scalar = glbChiNP(test_values, NULL, GLB_ALL);

      /* chi^2 for polarized vector DM fit */
      dm_set_dm_type(DM_VECTOR_POLARIZED);
      glbSetOscParamByName(test_values, 1., "ABS_XI_3");
      chi2_vector_pol = glbChiNP(test_values, NULL, GLB_ALL);

      /* chi^2 for unpolarized vector DM fit */
      dm_set_dm_type(DM_VECTOR_UNPOLARIZED);
      chi2_vector_unpol = glbChiNP(test_values, NULL, GLB_ALL);

      /* print output */
      printf("%15.10g  %15.10g %15.10g %15.10g\n",
             y_dm, chi2_scalar, chi2_vector_pol, chi2_vector_unpol);
    }
  }


  /* Destroy parameter vector(s) */
  glbFreeProjection(proj);
  glbFreeParams(input_errors);
  glbFreeParams(true_values);
  glbFreeParams(test_values); 
  
  exit(0);
}
