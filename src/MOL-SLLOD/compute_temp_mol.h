/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(temp/mol,ComputeTempMol);
// clang-format on
#else

#ifndef LMP_COMPUTE_TEMP_MOL_H
#define LMP_COMPUTE_TEMP_MOL_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeTempMol : public Compute {
 public:
  ComputeTempMol(class LAMMPS *, int, char **);
  ~ComputeTempMol() override;
  void init() override;
  void setup() override;
  double compute_scalar() override;
  void compute_vector() override;

  double memory_usage() override;

  // TODO(SS): centralise vcm_compute() to fix property/mol?
  void vcm_compute(double *ke_singles = nullptr);

 protected:
  char *id_molprop;
  class FixPropertyMol *molprop;

 private:
  int nmax;
  double adof, cdof, tfactor;

  void dof_compute();
  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
