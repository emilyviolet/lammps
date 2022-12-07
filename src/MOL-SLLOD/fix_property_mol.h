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

#ifdef FIX_CLASS
// clang-format off
FixStyle(property/mol,FixPropertyMol);
// clang-format on
#else

#ifndef LMP_FIX_PROPERTY_MOL_H
#define LMP_FIX_PROPERTY_MOL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPropertyMol : public Fix {
 public:
  FixPropertyMol(class LAMMPS *, int, char **);

  ~FixPropertyMol() override;
  int setmask() override;
  void init() override;
  void setup_pre_force(int) override;
  void setup_pre_force_respa(int, int) override;
  double memory_usage() override;
  double compute_array(int, int) override;

  // Calculate nmolecule and grow permolecule vectors/arrays as needed.
  // Return true if max. mol id changed.
  bool grow_permolecule(int=0);

  double *mass;           // per molecule mass
  double ke_singles[6];   // Vector components of kinetic energy
  double **com;           // per molecule center of mass in unwrapped coords
  double **vcm;           // per molecule center of mass velocity
  bigint mass_step;       // last step where mass was updated
  bigint com_step;        // last step where com was updated
  bigint vcm_step;        // last step where vcm was updated

  tagint molmax;          // Max. molecule id

  int dynamic_group;      // 1 = group is dynamic (nmolecule could change)
  int dynamic_mols;       // 1 = number of molecules could change during run

  bigint count_step;      // Last step where count_molecules was called
  tagint nmolecule;       // Number of molecules in the group
  void count_molecules();
  void mass_compute();
  void com_compute();
  void vcm_compute();

  void request_com();       // Request that CoM be allocated (implies mass)
  void request_vcm();    // Request that VCM be allocated (implies mass). Optional flag to calculate KE components
  void request_mass();      // Request that mass be allocated

 protected:
  tagint nmax;            // length of permolecule arrays the last time they grew

  double *massproc, **comproc, **vcmproc;
  int mass_flag, com_flag, vcm_flag, ke_singles_flag;    // 1 if mass/com can be computed

};

}    // namespace LAMMPS_NS

#endif
#endif
