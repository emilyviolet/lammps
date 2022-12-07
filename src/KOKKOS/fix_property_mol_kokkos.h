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
FixStyle(property/mol/kk,FixPropertyMolKokkos<LMPDeviceType>);
FixStyle(property/mol/kk/device,FixPropertyMolKokkos<LMPDeviceType>);
FixStyle(property/mol/kk/host,FixPropertyMolKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_PROPERTY_MOL_KOKKOS_H
#define LMP_FIX_PROPERTY_MOL_KOKKOS_H

#include "fix_property_mol.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {
struct TagFixPropertyMol_mass{};
struct TagFixPropertyMol_com{};
struct TagFixPropertyMol_count{};

template<class DeviceType>
class FixPropertyMolKokkos : public FixPropertyMol {
 public:
  typedef ArrayTypes<DeviceType> AT;
  FixPropertyMolKokkos(class LAMMPS *, int, char **);

  ~FixPropertyMolKokkos() override;
  int setmask() override;
  void init() override;
  void setup_pre_force(int) override;
  void setup_pre_force_respa(int, int) override;
  double memory_usage() override;
  double compute_array(int, int) override;

  struct PerMoleculeKK {
    std::string name;     // Identifier
    void *h_address;      // Host data address
    void *view_address;   // Address of Kokkos View
    int datatype;         // INT or DOUBLE
    int cols;             // number of columns (0 for vectors)
  };

  // Calculate nmolecule and grow permolecule vectors/arrays as needed.
  // Return true if max. mol id changed.
  bool grow_permolecule(int=0);

  //double* mass;           // per molecule mass 
  //double** com;           // per molecule center of mass in unwrapped coords
  double **vcm;             // Placeholder until I update the base class
  typename AT::tdual_float_1d k_mass;           // per molecule mass view
  typename AT::tdual_x_array k_com;           // per molecule center of mass view in unwrapped coords
  typename AT::tdual_v_array k_vcm;           // per molecule center of mass velocity
  //bigint mass_step;       // last step where mass was updated
  //bigint com_step;        // last step where com was updated

  //tagint molmax;          // Max. molecule id

  //int dynamic_group;      // 1 = group is dynamic (nmolecule could change)
  //int dynamic_mols;       // 1 = number of molecules could change during run

  //bigint count_step;      // Last step where count_molecules was called
  //tagint nmolecule;       // Number of molecules in the group
  void count_molecules();
  void mass_compute();
  void com_compute();

  void request_com();     // Request that CoM be allocated (implies mass)
  void request_vcm();     // Request that VCM be allocated (implies mass)
  void request_mass();    // Request that mass be allocated

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPropertyMol_mass, const int& i) const;
  void operator()(TagFixPropertyMol_com, const int& i) const;
  void operator()(TagFixPropertyMol_count, const int& i) const;

 protected:
  //tagint nmax;            // length of permolecule arrays the last time they grew
  int vcm_flag;

  //double *massproc, **comproc;
  double **vcmproc;     // Placeholder until I add this to the base class
  typename AT::tdual_float_1d k_massproc;
  typename AT::tdual_x_array k_comproc;
  typename AT::tdual_v_array k_vcmproc;
  typename AT::t_float_1d d_massproc;
  typename AT::t_x_array d_comproc;
  typename AT::t_v_array d_vcmproc;

  typename AT::t_x_array atom_x;
  typename AT::t_float_1d atom_rmass;
  typename AT::t_float_1d atom_mass;
  typename AT::t_int_1d atom_type;
  typename AT::t_int_1d atom_molID;
  typename AT::t_int_1d atom_mask;

  // Scatter views to use when calculating the per-molecule arrays across multiple threads
  Kokkos::Experimental::ScatterView<double*, typename AT::t_float_1d::array_layout> dup_massproc;
  Kokkos::Experimental::ScatterView<double*, typename AT::t_float_1d::array_layout> dup_comproc;

 private:
};

}    // namespace LAMMPS_NS

#endif
#endif
