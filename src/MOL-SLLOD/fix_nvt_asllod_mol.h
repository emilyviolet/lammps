/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(nvt/a-sllod/mol,FixNVTAsllodMol);
// clang-format on
#else

#ifndef LMP_FIX_NVT_ASLLOD_MOL_H
#define LMP_FIX_NVT_ASLLOD_MOL_H

#include "fix_nh.h"

namespace LAMMPS_NS {

class FixNVTAsllodMol : public FixNH {
 public:
  FixNVTAsllodMol(class LAMMPS *, int, char **);
  ~FixNVTAsllodMol();
  void post_constructor() override;

  void init() override;

 protected:
  int molpropflag;    // 1 = molprop created by nvt/sllod/mol, 0 = user supplied
  char *id_molprop;   // Name of property/molecule fix
  class FixPropertyMol *molprop;

 private:
  void nh_v_temp() override;
  void nve_x() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
