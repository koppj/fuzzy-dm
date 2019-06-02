// GLoBES fuzzy dark matter oscillation engine.
// see fuzzy-dm.c for author, license, and citation information
#ifndef __FUZZY_DM_H
#define __FUZZY_DM_H

#include <gsl/gsl_matrix.h>
#include <globes/globes.h>

// Arrangement of oscillation parameters in glb_params data structure:
//   th12,    th13,    th23,    deltaCP,
//   dm21,    dm31
//   |\chi_{ee},   arg(\chi_{ee}),   ..., |\chi_{tt}|,  arg(\chi_{et}),
//           ...             ...           ...
//   |\chi_{te},   arg(\chi_{te}),   ..., |\chi_{tt}|,  arg(\chi_{tt}),
//   |\xi^0|, arg(\xi^0) ... |\xi^3|, arg(\xi^3)
// The last 8 parameters (corresponding to the DM polarization) will
// be ignored for scalar DM. For unpolarized vector DM, the norm of \xi
// acts as an additional scale factor for the neutrino--DM couplings;
// for polarized DM, the norm of \xi acts as a scale factor, and also
// \xi^0 and \xi^3 matter independently (the beam is taken to be oriented
// along the z-axis)

// Supported types of DM
enum dm_types { DM_SCALAR, DM_VECTOR_POLARIZED, DM_VECTOR_UNPOLARIZED };

// Names of oscillation parameters
extern char dm_param_strings[][64];

// Maximum number of neutrino species and related quantities
#define DM_MAX_FLAVORS   3
#define DM_MAX_ANGLES    ((DM_MAX_FLAVORS * (DM_MAX_FLAVORS-1))/2)
#define DM_MAX_PHASES    (((DM_MAX_FLAVORS-1)*(DM_MAX_FLAVORS-2))/2)
#define DM_MAX_PARAMS    (DM_MAX_ANGLES + DM_MAX_PHASES + DM_MAX_FLAVORS-1 \
                          + 1 + 1 + 2*DM_MAX_FLAVORS*DM_MAX_FLAVORS + 8)
                              // see explanations in dm_init_probability_engine()


// Function declarations
// ---------------------

#ifdef __cplusplus
extern "C" {
#endif

// dm.c
int dm_init_probability_engine(int _dm_type);
int dm_get_n_osc_params();
int dm_set_dm_type(int _dm_type);
int dm_get_dm_type();
const char *dm_get_param_name(const int i);
int dm_free_probability_engine();
int dm_set_oscillation_parameters(glb_params p, void *user_data);
int dm_get_oscillation_parameters(glb_params p, void *user_data);
int dm_filtered_probability_matrix_cd(double P[DM_MAX_FLAVORS][DM_MAX_FLAVORS],
      double E, double L, double V, double sigma, int cp_sign, void *user_data);
int dm_probability_matrix(double _P[3][3], int cp_sign, double E,
      int psteps, const double *length, const double *density,
      double filter_sigma, void *user_data);
int dm_probability_matrix_all(double P[DM_MAX_FLAVORS][DM_MAX_FLAVORS], int cp_sign,
      double E, int psteps, const double *length, const double *density,
      double filter_sigma, void *user_data);
int dm_probability_matrix_m_to_f(double P[DM_MAX_FLAVORS][DM_MAX_FLAVORS],
      int cp_sign, double E, int psteps, const double *length, const double *density,
      double filter_sigma, void *user_data);
int dm_filtered_probability_matrix_m_to_f(double P[DM_MAX_FLAVORS][DM_MAX_FLAVORS],
      double E, double L, double V, double sigma, int cp_sign, void *user_data);
gsl_matrix_complex *dm_get_U();
int dm_print_gsl_matrix_complex(gsl_matrix_complex *A);

// prem.c
int LoadPREMProfile(const char *prem_file);
double GetPREMDensity(double t, double L);
double GetAvgPREMDensity(double L_tot, double L1, double L2);
int GetPREM3LayerApprox(double L, int *n_layers, double *lengths,
                        double *densities);

#ifdef __cplusplus
}
#endif

#endif

