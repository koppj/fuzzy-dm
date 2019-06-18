// ----------------------------------------------------------------------------
// Probability engine for simulating neutrino--fuzzy dark matter interactions
// ----------------------------------------------------------------------------
// This code implemented couplings between neutrinos and Fuzzy Dark Matter.
// The Lagrangian is
//    L \supset - y <\phi> \bar{\nu^c} \nu                 (scalar DM)
// or
//    L \supset - y <\phi^\mu> \bar\nu \gamma_\mu \nu      (vector DM)
// Here, y can be an arbitrary complex matrix in flavor space; the expectation
// value of \phi is given by \sqrt{2 \rho_DM} / m_\phi, where m_\phi is
// the DM mass and \rho_DM is the local DM density ~0.3 GeV/cm^3.
// We define the effective coupling
//   \chi = y <\phi>
// for scalar DM. For vector DM, we write \phi^\mu = \phi_0 \xi^\mu (where
// \xi^\mu is the polarization vector) and define \chi = y <\phi_0>.
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
// Note: This file is written in C99. To compile in gcc use -std=c99 or
// -std=gnu99
// ----------------------------------------------------------------------------
// ChangeLog:
//   2019-05-24: - branched off snu.c
// ----------------------------------------------------------------------------
// Citation information:
//
//      @Article{Kopp:2006wp,
//        author    = "Kopp, Joachim",
//        title     = "{Efficient numerical diagonalization of hermitian
//                     $3 \times 3$ matrices}",
//        journal   = "Int. J. Mod. Phys.",
//        volume    = "C19",
//        year      = "2008",
//        pages     = "523-548",
//        eprint    = "physics/0610206",
//        archivePrefix = "arXiv",
//        doi       = "10.1142/S0129183108012303",
//        SLACcitation  = "%%CITATION = PHYSICS/0610206;%%",
//        note      = "Erratum ibid.\ {\bf C19} (2008) 845",
//        memo      = "Algorithms for fast diagonalization of 3x3 matrices
//          (used for <= 3 neutrino flavors)"
//      }
//
//      @article{Brdar:2017kbt,
//	    author         = "Brdar, Vedran and Kopp, Joachim and Liu, Jia and Prass,
//			      Pascal and Wang, Xiao-Ping",
//	    title          = "{Fuzzy dark matter and nonstandard neutrino
//			      interactions}",
//	    journal        = "Phys. Rev.",
//	    volume         = "D97",
//	    year           = "2018",
//	    number         = "4",
//	    pages          = "043001",
//	    doi            = "10.1103/PhysRevD.97.043001",
//	    eprint         = "1705.09455",
//	    archivePrefix  = "arXiv",
//	    primaryClass   = "hep-ph",
//	    reportNumber   = "MITP-17-037",
//	    SLACcitation   = "%%CITATION = ARXIV:1705.09455;%%"
//      }
//
// For the underlying physics, we recommend to also cite
//
//      @article{Berlin:2016woy,
//	    author         = "Berlin, Asher",
//	    title          = "{Neutrino Oscillations as a Probe of Light Scalar Dark
//			      Matter}",
//	    journal        = "Phys. Rev. Lett.",
//	    volume         = "117",
//	    year           = "2016",
//	    number         = "23",
//	    pages          = "231801",
//	    doi            = "10.1103/PhysRevLett.117.231801",
//	    eprint         = "1608.01307",
//	    archivePrefix  = "arXiv",
//	    primaryClass   = "hep-ph",
//	    SLACcitation   = "%%CITATION = ARXIV:1608.01307;%%"
//      }
//
//      @article{Krnjaic:2017zlz,
//	    author         = "Krnjaic, Gordan and Machado, Pedro A. N. and Necib, Lina",
//	    title          = "{Distorted neutrino oscillations from time varying cosmic
//			      fields}",
//	    journal        = "Phys. Rev.",
//	    volume         = "D97",
//	    year           = "2018",
//	    number         = "7",
//	    pages          = "075017",
//	    doi            = "10.1103/PhysRevD.97.075017",
//	    eprint         = "1705.06740",
//	    archivePrefix  = "arXiv",
//	    primaryClass   = "hep-ph",
//	    reportNumber   = "FERMILAB-PUB-17-136-PPD, MIT-CTP-4908",
//	    SLACcitation   = "%%CITATION = ARXIV:1705.06740;%%"
//      }
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <complex.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include "globes/globes.h"
#include "fuzzy-dm.h"

// Constants
#define GLB_V_FACTOR        7.5e-14    // Conversion factor for matter potentials
#define GLB_Ne_MANTLE       0.5        // Effective electron numbers for calculation
#define GLB_Ne_CORE         0.468      //   of MSW potentials
#define RHO_THRESHOLD       0.001      // The minimum matter density below which
                                       // vacuum algorithms are used
#define M_SQRT3  1.73205080756887729352744634151     // sqrt(3)

#define C1           GSL_COMPLEX_ONE
#define C0           GSL_COMPLEX_ZERO

// Macros
#define SQR(x)      ((x)*(x))                        // x^2
#define SQR_ABS(x)  (SQR(creal(x)) + SQR(cimag(x)))  // |x|^2
#define POW10(x)    (exp(M_LN10*(x)))                // 10^x
#define MIN(X,Y)    ( ((X) < (Y)) ? (X) : (Y) )
#define MAX(X,Y)    ( ((X) > (Y)) ? (X) : (Y) )
#define SIGN(a,b)   ( (b) > 0.0 ? (fabs(a)) : (-fabs(a)) )
#define KRONECKER(i,j)  ( (i)==(j) ? 1 : 0 )

// Fundamental oscillation parameters
static int dm_type   = -1;                    // type of DM (one of the dm_types constants)
static int n_flavors = 0;
static int n_params  = 0;
static int n_angles  = 0;
static int n_phases  = 0;
static double th[DM_MAX_FLAVORS+1][DM_MAX_FLAVORS+1];// Mixing angles
static double delta[DM_MAX_PHASES];           // Dirac CP phases
static double dmsq[DM_MAX_FLAVORS-1];         // Mass squared differences in vacuum
static double dmsq_eff[DM_MAX_FLAVORS-1];     // Mass squared differences in DM background
static double m_1;                            // lightest neutrino mass
static double m_dm;                           // DM mass
static double complex chi_dm[DM_MAX_FLAVORS][DM_MAX_FLAVORS];
                                              // effective DM field (\chi = y*<\phi>)
static double complex xi_dm[4];               // DM polarization vector (for vector DM)

// parameter names
char dm_param_strings[DM_MAX_PARAMS][64];

// Internal temporary variables
static gsl_matrix_complex *U=NULL; // The vacuum mixing matrix
static gsl_matrix_complex *H=NULL; // Neutrino Hamiltonian
static gsl_matrix_complex *Q=NULL; // Eigenvectors of Hamiltonian (= eff. mixing matrix)
static gsl_vector *lambda=NULL;    // Eigenvalues of Hamiltonian
static gsl_matrix_complex *S=NULL; // The neutrino S-matrix

static gsl_matrix_complex *H_template_1=NULL; // coefficient of 1/E term in Hamiltonian
static gsl_matrix_complex *H_template_2=NULL; // coefficient of E-indep. term in  Hamiltonian
static gsl_matrix_complex *S1=NULL, *T0=NULL; // Temporary matrix storage

static gsl_eigen_hermv_workspace *w=NULL;     // Workspace for eigenvector algorithm

extern int density_corr[];

// The order in which the rotation matrices corresponding to the different
// mixing angles are multiplied together (numbers are indices to th[][]
static int rotation_order[DM_MAX_ANGLES][2];

// Which rotation matrices contain the complex phases? Indices are to
// delta[], -1 indicates no complex phase in a particular matrix;
// phase_order[0] corresponds to the leftmost rotation matrix
static int phase_order[DM_MAX_ANGLES];


// ----------------------------------------------------------------------------
//     3 x 3   E I G E N S Y S T E M   F U N C T I O N S  (physics/0610206)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
static void zhetrd3(double complex A[3][3], double complex Q[3][3],
                    double d[3], double e[2])
// ----------------------------------------------------------------------------
// Reduces a hermitian 3x3 matrix to real tridiagonal form by applying
// (unitary) Householder transformations:
//            [ d[0]  e[0]       ]
//    A = Q . [ e[0]  d[1]  e[1] ] . Q^T
//            [       e[1]  d[2] ]
// The function accesses only the diagonal and upper triangular parts of
// A. The access is read-only.
// ---------------------------------------------------------------------------
{
  const int n = 3;
  double complex u[n], q[n];
  double complex omega, f;
  double K, h, g;
  
  // Initialize Q to the identitity matrix
#ifndef EVALS_ONLY
  for (int i=0; i < n; i++)
  {
    Q[i][i] = 1.0;
    for (int j=0; j < i; j++)
      Q[i][j] = Q[j][i] = 0.0;
  }
#endif

  // Bring first row and column to the desired form 
  h = SQR_ABS(A[0][1]) + SQR_ABS(A[0][2]);
  if (creal(A[0][1]) > 0)
    g = -sqrt(h);
  else
    g = sqrt(h);
  e[0] = g;
  f    = g * A[0][1];
  u[1] = conj(A[0][1]) - g;
  u[2] = conj(A[0][2]);
  
  omega = h - f;
  if (creal(omega) > 0.0)
  {
    omega = 0.5 * (1.0 + conj(omega)/omega) / creal(omega);
    K = 0.0;
    for (int i=1; i < n; i++)
    {
      f    = conj(A[1][i]) * u[1] + A[i][2] * u[2];
      q[i] = omega * f;                  // p
      K   += creal(conj(u[i]) * f);      // u* A u
    }
    K *= 0.5 * SQR_ABS(omega);

    for (int i=1; i < n; i++)
      q[i] = q[i] - K * u[i];
    
    d[0] = creal(A[0][0]);
    d[1] = creal(A[1][1]) - 2.0*creal(q[1]*conj(u[1]));
    d[2] = creal(A[2][2]) - 2.0*creal(q[2]*conj(u[2]));
    
    // Store inverse Householder transformation in Q
#ifndef EVALS_ONLY
    for (int j=1; j < n; j++)
    {
      f = omega * conj(u[j]);
      for (int i=1; i < n; i++)
        Q[i][j] = Q[i][j] - f*u[i];
    }
#endif

    // Calculate updated A[1][2] and store it in f
    f = A[1][2] - q[1]*conj(u[2]) - u[1]*conj(q[2]);
  }
  else
  {
    for (int i=0; i < n; i++)
      d[i] = creal(A[i][i]);
    f = A[1][2];
  }

  // Make (23) element real
  e[1] = cabs(f);
#ifndef EVALS_ONLY
  if (e[1] != 0.0)
  {
    f = conj(f) / e[1];
    for (int i=1; i < n; i++)
      Q[i][n-1] = Q[i][n-1] * f;
  }
#endif
}


// ----------------------------------------------------------------------------
static int zheevc3(double complex A[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues of a hermitian 3x3 matrix A using Cardano's
// analytical algorithm.
// Only the diagonal and upper triangular parts of A are accessed. The access
// is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The hermitian input matrix
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
{
  double m, c1, c0;
  
  // Determine coefficients of characteristic poynomial. We write
  //       | a   d   f  |
  //  A =  | d*  b   e  |
  //       | f*  e*  c  |
  double complex de = A[0][1] * A[1][2];                            // d * e
  double dd = SQR_ABS(A[0][1]);                                  // d * conj(d)
  double ee = SQR_ABS(A[1][2]);                                  // e * conj(e)
  double ff = SQR_ABS(A[0][2]);                                  // f * conj(f)
  m  = creal(A[0][0]) + creal(A[1][1]) + creal(A[2][2]);
  c1 = (creal(A[0][0])*creal(A[1][1])  // a*b + a*c + b*c - d*conj(d) - e*conj(e) - f*conj(f)
          + creal(A[0][0])*creal(A[2][2])
          + creal(A[1][1])*creal(A[2][2]))
          - (dd + ee + ff);
  c0 = creal(A[2][2])*dd + creal(A[0][0])*ee + creal(A[1][1])*ff
            - creal(A[0][0])*creal(A[1][1])*creal(A[2][2])
            - 2.0 * (creal(A[0][2])*creal(de) + cimag(A[0][2])*cimag(de));
                             // c*d*conj(d) + a*e*conj(e) + b*f*conj(f) - a*b*c - 2*Re(conj(f)*d*e)

  double p, sqrt_p, q, c, s, phi;
  p = SQR(m) - 3.0*c1;
  q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
  sqrt_p = sqrt(fabs(p));

  phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
  phi = (1.0/3.0) * atan2(sqrt(fabs(phi)), q);
  
  c = sqrt_p*cos(phi);
  s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

  w[1]  = (1.0/3.0)*(m - c);
  w[2]  = w[1] + s;
  w[0]  = w[1] + c;
  w[1] -= s;

  return 0;
}


// ----------------------------------------------------------------------------
static int zheevq3(double complex A[3][3], double complex Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a hermitian 3x3
// matrix A using the QL algorithm with implicit shifts, preceded by a
// Householder reduction to real tridiagonal form.
// The function accesses only the diagonal and upper triangular parts of A.
// The access is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The hermitian input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error (no convergence)
// ----------------------------------------------------------------------------
// Dependencies:
//   zhetrd3()
// ----------------------------------------------------------------------------
{
  const int n = 3;
  double e[3];                 // The third element is used only as temporary workspace
  double g, r, p, f, b, s, c;  // Intermediate storage
  double complex t;
  int nIter;
  int m;

  // Transform A to real tridiagonal form by the Householder method
  zhetrd3(A, Q, w, e);
  
  // Calculate eigensystem of the remaining real symmetric tridiagonal matrix
  // with the QL method
  //
  // Loop over all off-diagonal elements
  for (int l=0; l < n-1; l++)
  {
    nIter = 0;
    while (1)
    {
      // Check for convergence and exit iteration loop if off-diagonal
      // element e(l) is zero
      for (m=l; m <= n-2; m++)
      {
        g = fabs(w[m])+fabs(w[m+1]);
        if (fabs(e[m]) + g == g)
          break;
      }
      if (m == l)
        break;
      
      if (nIter++ >= 30)
        return -1;

      // Calculate g = d_m - k
      g = (w[l+1] - w[l]) / (e[l] + e[l]);
      r = sqrt(SQR(g) + 1.0);
      if (g > 0)
        g = w[m] - w[l] + e[l]/(g + r);
      else
        g = w[m] - w[l] + e[l]/(g - r);

      s = c = 1.0;
      p = 0.0;
      for (int i=m-1; i >= l; i--)
      {
        f = s * e[i];
        b = c * e[i];
        if (fabs(f) > fabs(g))
        {
          c      = g / f;
          r      = sqrt(SQR(c) + 1.0);
          e[i+1] = f * r;
          c     *= (s = 1.0/r);
        }
        else
        {
          s      = f / g;
          r      = sqrt(SQR(s) + 1.0);
          e[i+1] = g * r;
          s     *= (c = 1.0/r);
        }
        
        g = w[i+1] - p;
        r = (w[i] - g)*s + 2.0*c*b;
        p = s * r;
        w[i+1] = g + p;
        g = c*r - b;

        // Form eigenvectors
#ifndef EVALS_ONLY
        for (int k=0; k < n; k++)
        {
          t = Q[k][i+1];
          Q[k][i+1] = s*Q[k][i] + c*t;
          Q[k][i]   = c*Q[k][i] - s*t;
        }
#endif 
      }
      w[l] -= p;
      e[l]  = g;
      e[m]  = 0.0;
    }
  }

  return 0;
}


// ----------------------------------------------------------------------------
static int zheevh3(double complex A[3][3], double complex Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a hermitian 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors. However,
// if conditions are such that a large error in the results is to be
// expected, the routine falls back to using the slower, but more
// accurate QL algorithm. Only the diagonal and upper triangular parts of A need
// to contain meaningful values. Access to A is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The hermitian input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   zheevc3(), zhetrd3(), zheevq3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1: Simplified fallback condition --> speed-up
//   v1.0: First released version
// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
  double norm;          // Squared norm or inverse norm of current eigenvector
//  double n0, n1;        // Norm of first and second columns of A
  double error;         // Estimated maximum roundoff error
  double t, u;          // Intermediate storage
  int j;                // Loop counter
#endif

  // Calculate eigenvalues
  zheevc3(A, w);

#ifndef EVALS_ONLY
//  n0 = SQR(creal(A[0][0])) + SQR_ABS(A[0][1]) + SQR_ABS(A[0][2]);
//  n1 = SQR_ABS(A[0][1]) + SQR(creal(A[1][1])) + SQR_ABS(A[1][2]);
  
  t = fabs(w[0]);
  if ((u=fabs(w[1])) > t)
    t = u;
  if ((u=fabs(w[2])) > t)
    t = u;
  if (t < 1.0)
    u = t;
  else
    u = SQR(t);
  error = 256.0 * DBL_EPSILON * SQR(u);
//  error = 256.0 * DBL_EPSILON * (n0 + u) * (n1 + u);

  Q[0][1] = A[0][1]*A[1][2] - A[0][2]*creal(A[1][1]);
  Q[1][1] = A[0][2]*conj(A[0][1]) - A[1][2]*creal(A[0][0]);
  Q[2][1] = SQR_ABS(A[0][1]);

  // Calculate first eigenvector by the formula
  //   v[0] = conj( (A - w[0]).e1 x (A - w[0]).e2 )
  Q[0][0] = Q[0][1] + A[0][2]*w[0];
  Q[1][0] = Q[1][1] + A[1][2]*w[0];
  Q[2][0] = (creal(A[0][0]) - w[0]) * (creal(A[1][1]) - w[0]) - Q[2][1];
  norm    = SQR_ABS(Q[0][0]) + SQR_ABS(Q[1][0]) + SQR(creal(Q[2][0]));

  // If vectors are nearly linearly dependent, or if there might have
  // been large cancellations in the calculation of A(I,I) - W(1), fall
  // back to QL algorithm
  // Note that this simultaneously ensures that multiple eigenvalues do
  // not cause problems: If W(1) = W(2), then A - W(1) * I has rank 1,
  // i.e. all columns of A - W(1) * I are linearly dependent.
  if (norm <= error)
    return zheevq3(A, Q, w);
  else                      // This is the standard branch
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][0] = Q[j][0] * norm;
  }
  
  // Calculate second eigenvector by the formula
  //   v[1] = conj( (A - w[1]).e1 x (A - w[1]).e2 )
  Q[0][1]  = Q[0][1] + A[0][2]*w[1];
  Q[1][1]  = Q[1][1] + A[1][2]*w[1];
  Q[2][1]  = (creal(A[0][0]) - w[1]) * (creal(A[1][1]) - w[1]) - creal(Q[2][1]);
  norm     = SQR_ABS(Q[0][1]) + SQR_ABS(Q[1][1]) + SQR(creal(Q[2][1]));
  if (norm <= error)
    return zheevq3(A, Q, w);
  else
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][1] = Q[j][1] * norm;
  }
  
  // Calculate third eigenvector according to
  //   v[2] = conj(v[0] x v[1])
  Q[0][2] = conj(Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1]);
  Q[1][2] = conj(Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1]);
  Q[2][2] = conj(Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1]);
#endif

  return 0;
}


// ----------------------------------------------------------------------------
//                    I N T E R N A L   F U N C T I O N S
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
int dm_print_gsl_matrix(gsl_matrix *A)
// ----------------------------------------------------------------------------
// Print entries of a real GSL matrix in human-readable form
// ----------------------------------------------------------------------------
{
  int i, j;
  for (i=0; i < A->size1; i++)
  {
    for (j=0; j < A->size2; j++)
      printf("%12.6g   ", gsl_matrix_get(A, i, j));
    printf("\n");
  }

  return 0;
}


// ----------------------------------------------------------------------------
int dm_print_gsl_matrix_complex(gsl_matrix_complex *A)
// ----------------------------------------------------------------------------
// Print entries of a complex GSL matrix in human-readable form
// ----------------------------------------------------------------------------
{
  int i, j;
  for (i=0; i < A->size1; i++)
  {
    for (j=0; j < A->size2; j++)
    {
      printf("%12.6g +%12.6g*I   ", GSL_REAL(gsl_matrix_complex_get(A, i, j)),
             GSL_IMAG(gsl_matrix_complex_get(A, i, j)));
    } 
    printf("\n");
  }

  return 0;
}


// ----------------------------------------------------------------------------
int dm_init_probability_engine(int _dm_type)
// ----------------------------------------------------------------------------
// Allocates internal data structures for the probability engine.
// ----------------------------------------------------------------------------
// Return values:
//   > 0: number of oscillation parameters defined
//   < 0: error
// ----------------------------------------------------------------------------
{
  dm_set_dm_type(_dm_type);

  // Number of oscillation parameters
  n_flavors = 3;
  n_angles  = 3;
  n_phases  = 1;
  n_params  = n_angles                 // mixing angles
            + n_phases                 // phases
            + (n_flavors-1)            // mass squared differences
            + 1                        // lightest neutrino mass eigenvalue
            + 1                        // DM mass
            + 2*SQR(n_flavors)         // 3x3 complex effective DM field in mass basis
            + 2*SQR(n_flavors)         // 3x3 complex effective DM field in flavor basis
            + 8;                       // DM polarization 4-vector
  const int _rotation_order[][2] = { {2,3}, {1,3}, {1,2} };
  const int _phase_order[] = { -1, 0, -1 };

  dm_free_probability_engine();
  
  U = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  H = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  Q = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  lambda = gsl_vector_alloc(n_flavors);
  S = gsl_matrix_complex_calloc(n_flavors, n_flavors);
    
  H_template_1 = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  H_template_2 = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  S1 = gsl_matrix_complex_calloc(n_flavors, n_flavors);
  T0 = gsl_matrix_complex_calloc(n_flavors, n_flavors);

  w  = gsl_eigen_hermv_alloc(n_flavors);

  for (int i=0; i < n_angles; i++)
  {
    if (_rotation_order[i][0] < 1 || _rotation_order[i][0] > n_angles ||
        _rotation_order[i][1] < 1 || _rotation_order[i][1] > n_angles)
    {
      fprintf(stderr, "dm_init_probability_engine: Incorrect rotation order specification.\n");
      return -2;
    }
    if (_phase_order[i] >= n_phases)
    {
      fprintf(stderr, "dm_init_probability_engine: Incorrect phase order specification.\n");
      return -3;
    }
    rotation_order[i][0] = _rotation_order[i][0];
    rotation_order[i][1] = _rotation_order[i][1];
    phase_order[i]       = _phase_order[i];
  }


  // Define names of oscillation parameters
  sprintf(dm_param_strings[0], "%s", "TH12");    // Standard oscillation parameters
  sprintf(dm_param_strings[1], "%s", "TH13");
  sprintf(dm_param_strings[2], "%s", "TH23");
  sprintf(dm_param_strings[3], "%s", "DELTA_0");
  sprintf(dm_param_strings[4], "%s", "DM21");
  sprintf(dm_param_strings[5], "%s", "DM31");

  int k = 6;
  for (int i=4; i <= n_flavors; i++)            // Mass squared differences
    sprintf(dm_param_strings[k++], "DM%d1", i);

  sprintf(dm_param_strings[k++], "M1");         // Lightest neutrino mass eigenvalue
  sprintf(dm_param_strings[k++], "MDM");        // DM mass

  for (int i=0; i < n_flavors; i++)             // Effective DM field in mass basis
    for (int j=0; j < n_flavors; j++)
    {
      sprintf(dm_param_strings[k++], "ABS_CHI_%d%d", i+1, j+1);
      sprintf(dm_param_strings[k++], "ARG_CHI_%d%d", i+1, j+1);
    }

  const char *flavors[] = { "E", "MU", "TAU" };
  for (int i=0; i < n_flavors; i++)             // Effective DM field in flavor basis
    for (int j=0; j < n_flavors; j++)
    {
      sprintf(dm_param_strings[k++], "ABS_CHI_%s%s", flavors[i], flavors[j]);
      sprintf(dm_param_strings[k++], "ARG_CHI_%s%s", flavors[i], flavors[j]);
    }

  for (int i=0; i < 4; i++)                     // DM polarization
  {
    sprintf(dm_param_strings[k++], "ABS_XI_%d", i);
    sprintf(dm_param_strings[k++], "ARG_XI_%d", i);
  }

  if (k != n_params)
  {
    fprintf(stderr, "dm_init_probability_engine: n_params has an incorrect value (%d).\n",
            n_params);
    return -2;
  }

//  printf("Oscillation engine initialized for %d neutrino flavors\n", n_flavors);
//  printf("Oscillation parameters are:\n");
//  for (int i=0; i < n_params; i++)
//  {
//    printf("  %-20s", dm_param_strings[i]);
//    if (i % 4 == 3)  printf("\n");
//  }
//  printf("\n");

  return n_params;
}


// ----------------------------------------------------------------------------
int dm_get_n_osc_params()
// ----------------------------------------------------------------------------
// Returns the number of oscillation parameters for this oscillation engine.
// (This depends on the type of DM simulated.)
// ----------------------------------------------------------------------------
{
  return n_params;
}


// ----------------------------------------------------------------------------
int dm_set_dm_type(int _dm_type)
// ----------------------------------------------------------------------------
// Set the type of DM to one of the dm_types constants
// ----------------------------------------------------------------------------
{
  int initialized = (dm_type >= 0);
  int status = -2;

  switch (_dm_type)
  {
    case DM_SCALAR:
    case DM_VECTOR_POLARIZED:
    case DM_VECTOR_UNPOLARIZED:
      dm_type = _dm_type;
      status = 0;
      break;

      break;

    default:
      fprintf(stderr, "dm_init_probability_engine: unknown DM type: %d. "
                      "Using DM_SCALAR\n", _dm_type);
      dm_type = DM_SCALAR;
      status = -1;
  }

  // call dm_set_oscillation_parameters to force update of internal data structures
  if (initialized)
  {
    glb_params p = glbAllocParams();
    dm_get_oscillation_parameters(p, NULL);
    dm_set_oscillation_parameters(p, NULL);
    glbFreeParams(p);
  }

  return status;
}


// ----------------------------------------------------------------------------
int dm_get_dm_type()
// ----------------------------------------------------------------------------
// Return the type of DM (one of the dm_types constants)
// ----------------------------------------------------------------------------
{
  return dm_type;
}


// ----------------------------------------------------------------------------
const char *dm_get_param_name(const int i)
// ----------------------------------------------------------------------------
// Returns the name of the i-th oscillation parameter
// ----------------------------------------------------------------------------
{
  return dm_param_strings[i];
}


// ----------------------------------------------------------------------------
int dm_free_probability_engine()
// ----------------------------------------------------------------------------
// Destroys internal data structures of the probability engine.
// ----------------------------------------------------------------------------
{
  if (w !=NULL)     { gsl_eigen_hermv_free(w);      w  = NULL; }

  if (T0!=NULL)     { gsl_matrix_complex_free(T0);  T0 = NULL; }
  if (S1!=NULL)     { gsl_matrix_complex_free(S1);  S1 = NULL; }
  if (H_template_1!=NULL) { gsl_matrix_complex_free(H_template_1);  H_template_1 = NULL; }
  if (H_template_2!=NULL) { gsl_matrix_complex_free(H_template_2);  H_template_2 = NULL; }
  
  if (S!=NULL)      { gsl_matrix_complex_free(S);   S = NULL; }
  if (lambda!=NULL) { gsl_vector_free(lambda);      lambda = NULL; }
  if (Q!=NULL)      { gsl_matrix_complex_free(Q);   Q = NULL; }
  if (H!=NULL)      { gsl_matrix_complex_free(H);   H = NULL; }
  if (U!=NULL)      { gsl_matrix_complex_free(U);   U = NULL; }

  return 0;
}


// ----------------------------------------------------------------------------
int dm_set_oscillation_parameters(glb_params p, void *user_data)
// ----------------------------------------------------------------------------
// Sets the fundamental oscillation parameters and precomputes the mixing
// matrix and part of the Hamiltonian.
// ----------------------------------------------------------------------------
{
  gsl_matrix_complex *R  = gsl_matrix_complex_alloc(n_flavors, n_flavors); // rot. matrix
  gsl_matrix_complex *T  = gsl_matrix_complex_alloc(n_flavors, n_flavors); // temp. storage
  gsl_matrix_complex *M  = gsl_matrix_complex_alloc(n_flavors,n_flavors);  // eff. mass matrix
  gsl_matrix_complex *C  = gsl_matrix_complex_alloc(n_flavors,n_flavors);  // temp. matrix
  gsl_matrix *X          = gsl_matrix_alloc(n_flavors, n_flavors);         // Re(C) + temp.
  gsl_matrix *Y          = gsl_matrix_alloc(n_flavors, n_flavors);         // Im(C)
  gsl_matrix *W          = gsl_matrix_alloc(n_flavors, n_flavors);         // EVs of X
  gsl_matrix *V          = gsl_matrix_alloc(n_flavors, n_flavors);         // EVs of W^t.X.W
  gsl_vector *lambda     = gsl_vector_alloc(n_flavors);     // temporary eigenvalue storage
  gsl_eigen_hermv_workspace *wsH = gsl_eigen_hermv_alloc(n_flavors); // eigensystem workspace
  gsl_eigen_symmv_workspace *wsS = gsl_eigen_symmv_alloc(n_flavors); // eigensystem workspace
  double complex (*_H_template_1)[n_flavors]
    = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(H_template_1, 0, 0);
  double complex (*_H_template_2)[n_flavors]
    = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(H_template_2, 0, 0);
  double complex (*_R)[n_flavors]
    = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(R, 0, 0);
  double complex (*_T)[n_flavors]
    = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(T, 0, 0);
  double complex (*_M)[n_flavors]
                        = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(M, 0, 0);
  double complex (*_C)[n_flavors]
                        = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(C, 0, 0);
  double (*_X)[n_flavors] = (double (*)[n_flavors]) gsl_matrix_ptr(X, 0, 0);
  double (*_Y)[n_flavors] = (double (*)[n_flavors]) gsl_matrix_ptr(Y, 0, 0);
  double (*_W)[n_flavors] = (double (*)[n_flavors]) gsl_matrix_ptr(W, 0, 0);
  double *_lambda = gsl_vector_ptr(lambda, 0);
  int i, j, k;
  int status = 0;

  gsl_matrix_complex_set_zero(H_template_1);
  gsl_matrix_complex_set_zero(H_template_2);
  gsl_matrix_complex_set_zero(H);

  // Implement correlations between density parameters. This requires that under
  // all circumstances the scaling of the matter density is performed _after_
  // calling set_oscillation_parameters! At present, this works only with
  // the hybrid minimizer (GLB_MIN_POWELL)!
//  for (j=0; j < glb_num_of_exps; j++)
//    if (density_corr[j] != j)
//      glbSetDensityParams(p, glbGetDensityParams(p, density_corr[j]), j);

  // Copy oscillation parameters
  th[1][2] = glbGetOscParams(p, GLB_THETA_12);    // Standard parameters
  th[1][3] = glbGetOscParams(p, GLB_THETA_13);
  th[2][3] = glbGetOscParams(p, GLB_THETA_23);
  delta[0] = glbGetOscParams(p, GLB_DELTA_CP);
  dmsq_eff[0] = dmsq[0] = glbGetOscParams(p, GLB_DM_21);
  dmsq_eff[1] = dmsq[1] = glbGetOscParams(p, GLB_DM_31);

  k = 6;
  for (i=4; i <= n_flavors; i++)                // Mass squared differences
    dmsq_eff[i-2] = dmsq[i-2] = glbGetOscParams(p, k++);

  m_1  = glbGetOscParams(p, k++);               // Lightest neutrino mass eigenvalue
  m_dm = glbGetOscParams(p, k++);               // DM mass


  double complex chi_dm_mass[n_flavors][n_flavors];   // Effective DM field in mass basis
  int chi_dm_mass_given = 0;
  for (i=0; i < n_flavors; i++)
  {
    for (j=0; j < n_flavors; j++)
    {
      chi_dm_mass[i][j] = glbGetOscParams(p,k) * cexp(I*glbGetOscParams(p,k+1));
      if (chi_dm_mass[i][j] != 0.)
        chi_dm_mass_given = 1;
      k += 2;
    }
  }

  double complex chi_dm_flavor[n_flavors][n_flavors]; // Effective DM field in flavor basis
  int chi_dm_flavor_given = 0;
  for (i=0; i < n_flavors; i++)
  {
    for (j=0; j < n_flavors; j++)
    {
      chi_dm_flavor[i][j] = glbGetOscParams(p,k) * cexp(I*glbGetOscParams(p,k+1));
      if (chi_dm_flavor[i][j] != 0.)
        chi_dm_flavor_given = 1;
      k += 2;
    }
  }

  if (chi_dm_mass_given && chi_dm_flavor_given)
  {
    fprintf(stderr, "dm_set_oscillation_parameters: warning: DM couplings "
                    "given both in mass and flavor basis.\n");
  }

  for (i=0; i < 4; i++)                         // DM polarization
  {
    xi_dm[i] = glbGetOscParams(p,k) * cexp(I*glbGetOscParams(p,k+1));
    k += 2;
  }


  // Multiply rotation matrices to define PMNS matrix
  gsl_matrix_complex_set_identity(U);
  for (i=0; i < n_angles; i++)
  {
    int u = rotation_order[i][0] - 1;
    int v = rotation_order[i][1] - 1;
    double complex c = cos(th[u+1][v+1]);
    double complex s = sin(th[u+1][v+1]);
    if (phase_order[i] >= 0)
      s *= cexp(-I * delta[phase_order[i]]);

    gsl_matrix_complex_set_identity(R);
    _R[u][u] = c;
    _R[v][v] = c;
    _R[u][v] = s;
    _R[v][u] = -conj(s);

    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, C1, U, R, C0, T);   // T = U.R
    gsl_matrix_complex_memcpy(U, T);                               // U = T
  }


  // Scalar DM: re-diagonalize mass matrix to obtain effective mixing parameters
  // ---------------------------------------------------------------------------
  if (dm_type == DM_SCALAR)
  {
    // Rotate chi_dm_mass into flavor basis, add contributions given in the flavor basis
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        _T[i][j] = chi_dm_mass[i][j];
    gsl_blas_zgemm(CblasNoTrans, CblasTrans,   C1, T, U, C0, C);   // C = T.U^t
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, C1, U, C, C0, T);   // T = U.C
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        chi_dm[i][j] = _T[i][j] + chi_dm_flavor[i][j];

    // check symmetry of DM coupling matrix
    for (i=1; i < n_flavors; i++)
      for (j=i+1; j < n_flavors; j++)
      {
        if (cabs(chi_dm[i][j] - chi_dm[j][i]) / cabs(chi_dm[i][j] + chi_dm[j][i]) > 1e-12)
        {
//          fprintf(stderr, "dm_set_oscillation_parameters: warning: non-symmetric "
//                          "coupling matrix will be symmetrized.\n");
          chi_dm[i][j] = chi_dm[j][i] = 0.5 * (chi_dm[i][j] + chi_dm[j][i]);
        }
      }

    // Define modified mass matrix M by starting from the standard mass matrix in the
    // mass basis, transforming to the flavor basis using the vacuum PMNS matrix,
    // and adding the DM terms.
    // To diagonalize the complex symmetric matrix M, we use the following procedure:
    //   - diagonalize M^+.M  -> diag(lambda) = R^+ . M^+.M . R
    //   - compute C = R^t . M . R, which is complex symmetric
    //   - we see that C^+.C = R^+.M^+.M.R = diag(lambda). Writing C = X + i*Y
    //     (with real X and Y), we have C^+.C = X^2 + Y^2 + i * (X.Y - Y.X).
    //     Since C^+.C is real, we must have [X,Y]=0, so X and Y can be
    //     diagonalized simultaneously. Compute this diagonalization:
    //       W^t . X . W = diag(m)
    //     In case X has two or more degenerate eigenvalues, we still need to
    //     diagonalize W^t . Y . W in the degenerate subspaces (but not outside in case Y
    //     has several identical eigenvalues belonging to different degenerate
    //     subspaces of X): V^t . W^t . Y . W . V = diag(m') [in degenerate subspaces of X].
    //   - M is then diagonalized by R.W according to
    //       V^t . W^t . R^t . M . R . W . V = diag(m)
    //   - Thus, the PMNS matrix is U = R.W.V
    //   - We finally need to determine the ordering of the eigenvalues that
    //     corresponds to any row/column permutations in U. This is achieved
    //     by re-diagonalizing M.
    gsl_matrix_complex_set_zero(M);
    _M[0][0] = m_1;                                                // M = m_\nu
    for (i=1; i < n_flavors; i++)
      _M[i][i] = sqrt(dmsq[i-1] + m_1);
    gsl_blas_zgemm(CblasConjTrans, CblasTrans,     C1, M, U, C0, T); // T = M^+ . U^t
    gsl_blas_zgemm(CblasConjTrans, CblasConjTrans, C1, T, U, C0, M); // M = T^+ . U^+
      // What we actually would like to compute is U^* . M . U^+. As BLAS does
      // not support complex conjugation in matrix multiplication, we instead compute
      // (U^* . M)^+ first, and take the Hermitian conjugate again in the 2nd step
    for (i=0; i < n_flavors; i++)                                  // M = M + \chi
      for (j=0; j < n_flavors; j++)
        _M[i][j] += chi_dm[i][j];

    gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, C1, M, M, C0, T); // T = M^+.M
    gsl_eigen_hermv(T, lambda, R, wsH);                            // (l,R) = eigensystem(T)
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, C1, M, R, C0, T);   // T = M.R
    gsl_blas_zgemm(CblasTrans,   CblasNoTrans, C1, R, T, C0, C);   // C = R^t.T

    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
      {
        _X[i][j] = creal(_C[i][j]);                                // X = Re(C)
        _Y[i][j] = cimag(_C[i][j]);                                // Y = Im(C)
      }
    gsl_eigen_symmv(X, lambda, W, wsS);                            // (l,W) = eigensystem(X)
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., Y, W, 0., X);   // X = Y . W
    gsl_blas_dgemm(CblasTrans,   CblasNoTrans, 1., W, X, 0., Y);   // Y = W^t . X

    double norm_X = 0.0;                                           // compute 1-norm of X
    for (i=0; i < n_flavors; i++)
      norm_X += fabs(_lambda[i]);
    int deg[n_flavors];
    for (i=0; i < n_flavors; i++)                                  // find degenerate EVs
    {
      deg[i] = 0;
      for (j=i+1; j < n_flavors; j++)
        if (norm_X > 0.  &&  fabs(_lambda[j] - _lambda[i]) / norm_X < 1e-12)
          deg[i] = deg[j] = 1;
        else
          deg[j] = 0;

      if (deg[i])                                                  // degenerate EV found:
      {                                                            // -> diag. Y in that space
        for (i=0; i < n_flavors; i++)
          for (j=0; j < n_flavors; j++)
            if (deg[i] && deg[j])                                  // X = Y (in degenerate
              _X[i][j] = _Y[i][j];                                 //   subspace)
            else
              _X[i][j] = (i==j) ? (i+1) * norm_X : 0.0;            // make X diagonal and
                                                                   //   non-degenerate outside
                                                                   //   the subspace
        gsl_eigen_symmv(X, lambda, V, wsS);                        // (l,V) = eigensystem(X)
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., W, V, 0., X); // X = W . V
        gsl_matrix_memcpy(W, X);                                   // W = X

        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., Y, V, 0., X); // X = Y . V
        gsl_blas_dgemm(CblasTrans,   CblasNoTrans, 1., W, X, 0., Y); // Y = V^t . X

        deg[i] = 0;                                                // mark this EV as done
      }
    }

    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        _T[i][j] = _W[i][j];                                       // T = W + 0*i
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, C1, R, T, C0, U);   // U = R . T

    // Re-diagonalize M using U to determine the correct ordering of eigenvalues
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, C1, M, U, C0, T);   // T = M . U
    gsl_blas_zgemm(CblasTrans,   CblasNoTrans, C1, U, T, C0, M);   // M = U^t . T
    for (i=1; i < n_flavors; i++)
      dmsq_eff[i-1] = _M[i][i]*conj(_M[i][i]) - _M[0][0]*conj(_M[0][0]);

    // Pre-compute energy independent matrix H0 * E
    for (i=1; i < n_flavors; i++)
      _H_template_1[i][i] = 0.5*dmsq_eff[i-1];
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, C1, H_template_1, U, C0, T); // T=H0.U^+
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans,   C1, U, T, C0, H_template_1); // H0=U.T
  }


  // Polarized Vector DM
  // -------------------
  else if (dm_type == DM_VECTOR_POLARIZED)
  {
    // Rotate chi_dm_mass into flavor basis, add contributions given in the flavor basis
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        _T[i][j] = chi_dm_mass[i][j];
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, C1, T, U, C0, C);   // C = T.U^+
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans,   C1, U, C, C0, T);   // T = U.C
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        chi_dm[i][j] = _T[i][j] + chi_dm_flavor[i][j];

    // check hermiticity of DM couplings
    for (i=1; i < n_flavors; i++)
      for (j=i+1; j < n_flavors; j++)
      {
        if (cabs(chi_dm[i][j] - conj(chi_dm[j][i]))
               / (cabs(chi_dm[i][j]) + cabs(chi_dm[j][i])) > 1e-12)
        {
//          fprintf(stderr, "dm_set_oscillation_parameters: warning: non-hermitian "
//                          "coupling matrix will be made hermitian.\n");
          chi_dm[i][j] = 0.5 * (chi_dm[i][j] + conj(chi_dm[j][i]));
          chi_dm[j][i] = conj(chi_dm[i][j]);
        }
      }

    // check normalization of polarization vector
    double xi_norm_sq = 0.0;
    for (i=1; i < 4; i++)
      xi_norm_sq += SQR(creal(xi_dm[i])) + SQR(cimag(xi_dm[i]));
//    if (fabs(xi_norm_sq) - 1. > 1e-12)
//    {
//      fprintf(stderr, "dm_set_oscillation_parameters: warning: polarization "
//                      "vector not normalized.\n");
//    }

    // coefficient of 1/E in the Hamiltonian
    for (i=1; i < n_flavors; i++)
      _H_template_1[i][i] = 0.5*dmsq[i-1];
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, C1, H_template_1, U, C0, T); // T=H0.U^+
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, C1, U, T, C0, H_template_1);   // H0=U.T
    for (i=0; i < n_flavors; i++)                                             // H0+=chi^+ chi
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _H_template_1[i][j] += conj(chi_dm[k][i]) * chi_dm[k][j] * xi_norm_sq;

    // energy-independent terms in the Hamiltonian
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        _H_template_2[i][j] = -chi_dm[i][j] * (xi_dm[0] - xi_dm[3]);
  }


  // Unpolarized Vector DM: same as polarized vector DM, but without the p.\xi term
  // ------------------------------------------------------------------------------
  else if (dm_type == DM_VECTOR_UNPOLARIZED)
  {
    // Rotate chi_dm_mass into flavor basis, add contributions given in the flavor basis
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        _T[i][j] = chi_dm_mass[i][j];
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, C1, T, U, C0, C);   // C = T.U^+
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans,   C1, U, C, C0, T);   // T = U.C
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        chi_dm[i][j] = _T[i][j] + chi_dm_flavor[i][j];

    // check hermiticity of DM couplings
    for (i=1; i < n_flavors; i++)
      for (j=i+1; j < n_flavors; j++)
      {
        if (cabs(chi_dm[i][j] - conj(chi_dm[j][i]))
               / (cabs(chi_dm[i][j]) + cabs(chi_dm[j][i])) > 1e-12)
        {
//          fprintf(stderr, "dm_set_oscillation_parameters: warning: non-hermitian "
//                          "coupling matrix will be made hermitian.\n");
          chi_dm[i][j] = 0.5 * (chi_dm[i][j] + conj(chi_dm[j][i]));
          chi_dm[j][i] = conj(chi_dm[i][j]);
        }
      }

    // check normalization of polarization vector
    double xi_norm_sq = 0.0;
    for (i=1; i < 4; i++)
      xi_norm_sq += SQR(creal(xi_dm[i])) + SQR(cimag(xi_dm[i]));
//    if (fabs(xi_norm_sq) - 1. > 1e-12)
//    {
//      fprintf(stderr, "dm_set_oscillation_parameters: warning: polarization "
//                      "vector not normalized.\n");
//    }

    // coefficient of 1/E in the Hamiltonian
    for (i=1; i < n_flavors; i++)
      _H_template_1[i][i] = 0.5*dmsq[i-1];
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, C1, H_template_1, U, C0, T); // T=H0.U^+
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, C1, U, T, C0, H_template_1);   // H0=U.T
    for (i=0; i < n_flavors; i++)                                             // H0+=chi^+ chi
      for (j=0; j < n_flavors; j++)
        for (k=0; k < n_flavors; k++)
          _H_template_1[i][j] += conj(chi_dm[k][i]) * chi_dm[k][j] * xi_norm_sq;
  }

  // else: fall back to the SM
  else
  {
    fprintf(stderr, "dm_set_oscillation_parameters: unknown DM model: %d. "
                    "Falling back to SM.\n", dm_type);

    // coefficient of 1/E in the Hamiltonian
    for (i=1; i < n_flavors; i++)
      _H_template_1[i][i] = 0.5*dmsq[i-1];
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, C1, H_template_1, U, C0, T); // T=H0.U^+
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, C1, U, T, C0, H_template_1);   // H0=U.T
  }


  // Clean up
  gsl_eigen_symmv_free(wsS);
  gsl_eigen_hermv_free(wsH);
  gsl_vector_free(lambda);
  gsl_matrix_free(V);
  gsl_matrix_free(W);
  gsl_matrix_free(Y);
  gsl_matrix_free(X);
  gsl_matrix_complex_free(C);
  gsl_matrix_complex_free(M);
  gsl_matrix_complex_free(T);
  gsl_matrix_complex_free(R);

  return status;
}


// ----------------------------------------------------------------------------
int dm_get_oscillation_parameters(glb_params p, void *user_data)
// ----------------------------------------------------------------------------
// Returns the current set of oscillation parameters.
// If *user_data == DM_FLAVOR_basis, the function fills the
// ABS/ARG_CHI_ALPHABETA entries of p (DM coupling in flavor basis).  If
// *user_data == DM_MASS_basis, the function fills the ABS/ARG_CHI_IJ entries
// of p (DM coupling in mass basis).  The default is DM_FLAVOR_BASIS.
// By filling only one or the other set of equivalent parameters, we ensure
// that the results from dm_get_oscillation_parameters can be used directly
// in dm_set_oscillation_parameters.
// ----------------------------------------------------------------------------
{
  int i, j, k;

  int mode = DM_FLAVOR_BASIS;
  if (user_data)
    mode = *((int *) user_data);

  glbDefineParams(p, th[1][2], th[1][3], th[2][3], delta[0], dmsq[0], dmsq[1]);
                                                // standard oscillation parameters
  
  k = 6;
  for (i=4; i <= n_flavors; i++)                // Mass squared differences
    glbSetOscParams(p, dmsq[i-2], k++);

  glbSetOscParams(p, m_1, k++);                 // Lightest neutrino mass eigenvalue
  glbSetOscParams(p, m_dm, k++);                // DM mass

  for (i=1; i <= n_flavors; i++)                // Sterile mixing angles
    for (j=MAX(i+1,4); j <= n_flavors; j++)
      glbSetOscParams(p, th[i][j], k++);

  for (i=1; i <= n_phases-1; i++)               // Sterile phases
    glbSetOscParams(p, delta[i], k++);

 
  if (mode == DM_MASS_BASIS)                    // Effective DM field in the mass basis
  {
    gsl_matrix_complex *T  = gsl_matrix_complex_alloc(n_flavors, n_flavors); // temp. storage
    gsl_matrix_complex *C  = gsl_matrix_complex_alloc(n_flavors, n_flavors); // temp. storage
    double complex (*_T)[n_flavors]
      = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(T, 0, 0);

    // Transform effective DM field into the mass basis (for ABS/ARG_CHI_IJ)
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        _T[i][j] = chi_dm[i][j];
    gsl_blas_zgemm(CblasNoTrans,   CblasNoTrans, C1, T, U, C0, C);   // C = T.U
    gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, C1, U, C, C0, T);   // T = U^+.C

    for (i=0; i < n_flavors; i++) 
      for (j=0; j < n_flavors; j++)
      {
        glbSetOscParams(p, cabs(_T[i][j]), k);
        glbSetOscParams(p, carg(_T[i][j]), k+1);
        k += 2;
      }
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
      {
        glbSetOscParams(p, 0., k);
        glbSetOscParams(p, 0., k+1);
        k += 2;
      }

    gsl_matrix_complex_free(C);
    gsl_matrix_complex_free(T);
  }
  else                                          // Effective DM field in the flavor basis
  {
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
      {
        glbSetOscParams(p, 0., k);
        glbSetOscParams(p, 0., k+1);
        k += 2;
      }
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
      {
        glbSetOscParams(p, cabs(chi_dm[i][j]), k);
        glbSetOscParams(p, carg(chi_dm[i][j]), k+1);
        k += 2;
      }
  }

  for (i=0; i < 4; i++)                         // DM polarization
  {
    glbSetOscParams(p, cabs(xi_dm[i]), k);
    glbSetOscParams(p, carg(xi_dm[i]), k+1);
    k += 2;
  }

  return 0;
}


// ----------------------------------------------------------------------------
int dm_hamiltonian_cd(double E, double rho, double Ne, int cp_sign)
// ----------------------------------------------------------------------------
// Calculates the Hamiltonian for neutrinos (cp_sign=1) or antineutrinos
// (cp_sign=-1) with energy E, propagating in matter of density rho
// (> 0 even for antineutrinos) and stores the result in H. Ne is the
// electron/proton fraction in matter (1 for solar matter, about 0.5 for Earth
// matter)
// ----------------------------------------------------------------------------
{
  double inv_E = 1.0 / E;
  double Ve = cp_sign * rho * (GLB_V_FACTOR * Ne); // Matter potential
  double Vn = cp_sign * rho * (GLB_V_FACTOR * (1.0 - Ne) / 2.0);

  double complex (*_H)[n_flavors]
    = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(H, 0, 0);
  double complex (*_H_template_1)[n_flavors]
    = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(H_template_1, 0, 0);
  double complex (*_H_template_2)[n_flavors]
    = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(H_template_2, 0, 0);
  int i, j;

  if (cp_sign > 0)
  {
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        _H[i][j] = _H_template_1[i][j] * inv_E + _H_template_2[i][j];
  }
  else
  {
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        _H[i][j] = conj(_H_template_1[i][j] * inv_E
                     - _H_template_2[i][j]); // delta_CP -> -delta_CP
  }

  // Add standard matter potential \sqrt{2} G_F (N_e - N_n/2) for \nu_e and
  // - \sqrt{2} G_F N_n / 2 for \nu_\mu and \nu_\tau
  _H[0][0] = _H[0][0] + Ve - Vn;
  _H[1][1] = _H[1][1] - Vn;
  _H[2][2] = _H[2][2] - Vn;

  return 0;
}


// ----------------------------------------------------------------------------
int dm_S_matrix_cd(double E, double L, double rho, int cp_sign, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the S matrix for neutrino oscillations in matter of constant
// density.
// ----------------------------------------------------------------------------
// Parameters:
//   E: Neutrino energy
//   L: Baseline
//   rho: Matter density (must be > 0 even for antineutrinos)
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
//   user_data: User-defined parameters (used for instance to tell the
//            probability engine which experiment it is being run for)
// ----------------------------------------------------------------------------
{
  // Introduce some abbreviations
  double complex (*_S)[n_flavors] =(double complex (*)[n_flavors])gsl_matrix_complex_ptr(S,0,0);
  double complex (*_Q)[n_flavors] =(double complex (*)[n_flavors])gsl_matrix_complex_ptr(Q,0,0);
  double complex (*_T0)[n_flavors]=(double complex (*)[n_flavors])gsl_matrix_complex_ptr(T0,0,0);
  double *_lambda = gsl_vector_ptr(lambda,0);
  int status;
  int i, j, k;
  
  if (fabs(rho) < RHO_THRESHOLD)                   // Vacuum
  {
    // Use vacuum mixing angles and masses
    double inv_E = 0.5/E;
    _lambda[0] = 0.0;
    for (i=1; i < n_flavors; i++)
      _lambda[i] = dmsq_eff[i-1] * inv_E;
          // vanilla GLoBES uses the vacuum \Delta m^2 here (dmsq), but for scalar DM,
          // the effective \Delta m^2 may be different.

    if (cp_sign > 0)
      gsl_matrix_complex_memcpy(Q, U);
    else
    {
      double complex (*_U)[n_flavors]
        = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(U,0,0);
      for (i=0; i < n_flavors; i++)
        for (j=0; j < n_flavors; j++)
          _Q[i][j] = conj(_U[i][j]);
    }
  }
  else                                             // Matter
  {
    // Calculate neutrino Hamiltonian
    if ((status=dm_hamiltonian_cd(E, rho, GLB_Ne_MANTLE, cp_sign)) != 0)
      return status;
    
    // Calculate eigenvalues of Hamiltonian
    if (n_flavors == 3)
    {
      double complex (*_H)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(H,0,0);
      double complex (*_Q)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(Q,0,0);
      double *_lambda = gsl_vector_ptr(lambda,0);
      if ((status=zheevh3(_H, _Q, _lambda)) != 0)
        return status;
    }
    else
    {
      if ((status=gsl_eigen_hermv(H, lambda, Q, w)) != GSL_SUCCESS)
        return status;
    }
  }

  // Calculate S-Matrix in mass basis in matter ...
  double phase;
  gsl_matrix_complex_set_zero(S);
  for (i=0; i < n_flavors; i++)
  {
    phase    = -L * _lambda[i];
    _S[i][i] = cos(phase) + I*sin(phase);
  } 
  
  // ... and transform it to the flavour basis
  gsl_matrix_complex_set_zero(T0);
  double complex *p = &_T0[0][0];
  for (i=0; i < n_flavors; i++)              // T0 = S.Q^+
    for (j=0; j < n_flavors; j++)
    {
      for (int k=0; k < n_flavors; k++)
      {
        *p += ( creal(_S[i][k])*creal(_Q[j][k])+cimag(_S[i][k])*cimag(_Q[j][k]) )
                + I * ( cimag(_S[i][k])*creal(_Q[j][k])-creal(_S[i][k])*cimag(_Q[j][k]) );
      }
      p++;
    }
  gsl_matrix_complex_set_zero(S);
  p = &_S[0][0];
  for (i=0; i < n_flavors; i++)              // S = Q.T0
    for (j=0; j < n_flavors; j++)
    {
      for (k=0; k < n_flavors; k++)
      {
        *p += ( creal(_Q[i][k])*creal(_T0[k][j])-cimag(_Q[i][k])*cimag(_T0[k][j]) )
                + I * ( cimag(_Q[i][k])*creal(_T0[k][j])+creal(_Q[i][k])*cimag(_T0[k][j]) );
      }
      p++;
    }

  return 0;
}


// ----------------------------------------------------------------------------
int dm_filtered_probability_matrix_cd(double P[DM_MAX_FLAVORS][DM_MAX_FLAVORS],
        double E, double L, double rho, double sigma, int cp_sign, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the probability matrix for neutrino oscillations in matter
// of constant density, including a low pass filter to suppress aliasing
// due to very fast oscillations.
// ----------------------------------------------------------------------------
// Parameters:
//   P: Storage buffer for the probability matrix
//   E: Neutrino energy (in eV)
//   L: Baseline (in eV^-1)
//   rho: Matter density (must be > 0 even for antineutrinos)
//   sigma: Width of Gaussian filter (in GeV)
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
//   user_data: User-defined parameters (used for instance to tell the
//            probability engine which experiment it is being run for)
// ----------------------------------------------------------------------------
{
  // Introduce some abbreviations
  double complex (*_Q)[n_flavors]  = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(Q,0,0);
  double complex (*_T0)[n_flavors] = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(T0,0,0);
  double *_lambda = gsl_vector_ptr(lambda,0);
  int status;
  int i, j, k, l;

  // Vacuum: Use vacuum mixing angles and masses
  if (fabs(rho) < RHO_THRESHOLD)
  {
    double inv_E = 0.5/E;
    _lambda[0] = 0.0;
    for (i=1; i < n_flavors; i++)
      _lambda[i] = dmsq_eff[i-1] * inv_E;
          // vanilla GLoBES uses the vacuum \Delta m^2 here (dmsq), but for scalar DM,
          // the effective \Delta m^2 may be different.

    if (cp_sign > 0)
      gsl_matrix_complex_memcpy(Q, U);
    else
    {
      double complex (*_U)[n_flavors]
        = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(U,0,0);
      for (i=0; i < n_flavors; i++)
        for (j=0; j < n_flavors; j++)
          _Q[i][j] = conj(_U[i][j]);
    }
  }

  // Matter: Rediagonalize Hamiltonian
  else
  {
    // Calculate neutrino Hamiltonian
    if ((status=dm_hamiltonian_cd(E, rho, GLB_Ne_MANTLE, cp_sign)) != 0)
      return status;
    
    // Calculate eigenvalues and eigenvectors of Hamiltonian
    if (n_flavors == 3)
    {
      double complex (*_H)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(H,0,0);
      double complex (*_Q)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(Q,0,0);
      double *_lambda = gsl_vector_ptr(lambda,0);
      if ((status=zheevh3(_H, _Q, _lambda)) != 0)
        return status;
    }
    else
    {
      if ((status=gsl_eigen_hermv(H, lambda, Q, w)) != GSL_SUCCESS)
        return status;
    }
  }

  // Calculate probability matrix (see GLoBES manual for a discussion of the algorithm)
  double phase, filter_factor;
  double t = -0.5/1.0e-18 * SQR(sigma) / SQR(E);
  gsl_matrix_complex_set_zero(T0);
  for (i=0; i < n_flavors; i++)
    for (j=i+1; j < n_flavors; j++)
    {
      phase         = -L * (_lambda[i] - _lambda[j]);
      filter_factor = exp(t * SQR(phase));
      _T0[i][j]     = filter_factor * (cos(phase) + I*sin(phase));
    }

  for (k=0; k < n_flavors; k++)
    for (l=0; l < n_flavors; l++)
    {
      P[k][l] = 0.0;
      for (i=0; i < n_flavors; i++)
      {
        complex t = conj(_Q[k][i]) * _Q[l][i];
        for (j=i+1; j < n_flavors; j++)
          P[k][l] += 2.0 * creal(_Q[k][j] * conj(_Q[l][j]) * t * _T0[i][j]);
        P[k][l] += SQR_ABS(_Q[k][i]) * SQR_ABS(_Q[l][i]);
      }
    }
    
  return 0;
}


// ----------------------------------------------------------------------------
int dm_probability_matrix(double _P[3][3], int cp_sign, double E,
    int psteps, const double *length, const double *density,
    double filter_sigma, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the neutrino oscillation probability matrix for use by GLoBES.
// The problem is that GLoBES expects P to be a 3x3 matrix, so we compute the
// full matrix and then extract the upper left 3x3 submatrix.
// ----------------------------------------------------------------------------
{
  double P[DM_MAX_FLAVORS][DM_MAX_FLAVORS];
  int status;
  int i, j;

  status = dm_probability_matrix_all(P, cp_sign, E, psteps, length, density,
                                     filter_sigma, user_data);
  for (i=0; i < 3; i++)
    for (j=0; j < 3; j++)
      _P[j][i] = P[j][i];

  return status;
}


// ----------------------------------------------------------------------------
int dm_probability_matrix_all(double P[DM_MAX_FLAVORS][DM_MAX_FLAVORS], int cp_sign, double E,
    int psteps, const double *length, const double *density,
    double filter_sigma, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the neutrino oscillation probability matrix.
// ----------------------------------------------------------------------------
// Parameters:
//   P:       Buffer for the storage of the matrix
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
//   E:       Neutrino energy (in GeV)
//   psteps:  Number of layers in the matter density profile
//   length:  Lengths of the layers in the matter density profile in km
//   density: The matter densities in g/cm^3
//   filter_sigma: Width of low-pass filter or <0 for no filter
//   user_data: User-defined parameters (used for instance to tell the
//            probability engine which experiment it is being run for)
// ----------------------------------------------------------------------------
{
  int status;
  int i, j;

  // Convert energy to eV
  E *= 1.0e9;
  
  if (filter_sigma > 0.0)                     // With low-pass filter
  {
    if (psteps == 1)
      dm_filtered_probability_matrix_cd(P, E, GLB_KM_TO_EV(length[0]),
                                         density[0], filter_sigma, cp_sign, user_data);
    else
      return -1;
  }
  else                                        // Without low-pass filter
  {
    if (psteps > 1)
    {
      gsl_matrix_complex_set_identity(S1);                                 // S1 = 1
      for (i=0; i < psteps; i++)
      {
        status = dm_S_matrix_cd(E, GLB_KM_TO_EV(length[i]), density[i], cp_sign, user_data);
        if (status != 0)
          return status;
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, S, S1, // T0 = S.S1
                       GSL_COMPLEX_ZERO, T0);
        gsl_matrix_complex_memcpy(S1, T0);                                 // S1 = T0
      } 
      gsl_matrix_complex_memcpy(S, S1);                                    // S = S1
    }
    else
    {
      status = dm_S_matrix_cd(E, GLB_KM_TO_EV(length[0]), density[0], cp_sign, user_data);
      if (status != 0)
        return status;
    }

    double complex (*_S)[n_flavors]
      = (double complex (*)[n_flavors]) gsl_matrix_complex_ptr(S,0,0);
    for (i=0; i < n_flavors; i++)
      for (j=0; j < n_flavors; j++)
        P[j][i] = SQR_ABS(_S[i][j]);
  }

  return 0;
}

