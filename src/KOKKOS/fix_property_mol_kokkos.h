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
#include "kokkos_few.h"

namespace LAMMPS_NS {
struct TagFixPropertyMol_count{};

struct TagFixPropertyMol_massproc_zero{};
struct TagFixPropertyMol_comproc_zero{};
struct TagFixPropertyMol_vcmproc_zero{};

struct TagFixPropertyMol_mass_compute{};

//template<int RMASS>
struct TagFixPropertyMol_com_compute{};
struct TagFixPropertyMol_com_scale{};

struct TagFixPropertyMol_vcm_compute{};
struct TagFixPropertyMol_vcm_scale{};

template<class DeviceType>
class FixPropertyMolKokkos : public FixPropertyMol {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  FixPropertyMolKokkos(class LAMMPS *, int, char **);

  ~FixPropertyMolKokkos() override;
  int setmask() override;
  void init() override;
  void setup_pre_force(int) override;
  void setup_pre_force_respa(int, int) override;
  double memory_usage() override;
  double compute_array(int, int) override;

  // Calculate nmolecule and grow permolecule vectors/arrays as needed.
  // Return true if max. mol id changed.
  bool grow_permolecule(int=0);

  // Dual views
  typename DAT::tdual_float_1d k_mass;           // per molecule mass view
  typename DAT::tdual_x_array k_com;           // per molecule center of mass view in unwrapped coords
  typename DAT::tdual_v_array k_vcm;           // per molecule center of mass velocity
  typename DAT::tdual_float_1d k_ke_singles;    // kinetic energy tensor
  double *ke_singles;    // kinetic energy tensor (host)

  // Device views
  typename AT::t_float_1d d_mass;           // per molecule mass view
  typename AT::t_x_array d_com;           // per molecule center of mass view in unwrapped coords
  typename AT::t_v_array d_vcm;           // per molecule center of mass velocity
  typename AT::t_float_1d d_ke_singles;    // kinetic energy tensor

  void count_molecules();
  void mass_compute();
  void com_compute();
  void vcm_compute();

  void request_com();     // Request that CoM be allocated (implies mass)
  void request_vcm();     // Request that VCM be allocated (implies mass)
  void request_mass();    // Request that mass be allocated

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPropertyMol_count, const int&, tagint&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPropertyMol_massproc_zero, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPropertyMol_comproc_zero, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPropertyMol_vcmproc_zero, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPropertyMol_mass_compute, const int&) const;
  
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPropertyMol_com_compute, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPropertyMol_com_scale, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPropertyMol_vcm_compute, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPropertyMol_vcm_scale, const int&) const;


 protected:

  double *keproc;
  typename DAT::tdual_float_1d k_massproc;
  typename DAT::tdual_x_array k_comproc;
  typename DAT::tdual_v_array k_vcmproc;
  typename DAT::tdual_float_1d k_keproc;

  typename AT::t_float_1d d_massproc;
  typename AT::t_x_array d_comproc;
  typename AT::t_v_array d_vcmproc;
  //Few<double, 6> keproc; 
  typename AT::t_float_1d d_keproc;

  typename AT::t_x_array atom_x;
  typename AT::t_v_array atom_v;
  typename AT::t_float_1d atom_rmass;
  typename AT::t_float_1d atom_mass;
  typename AT::t_int_1d atom_type;
  typename AT::t_int_1d atom_molID;
  typename AT::t_int_1d atom_mask;
  typename AT::t_int_1d atom_image;

  // Variables for domain unwrapping
  Few<double, 6> h;
  Few<double, 3> prd;
  int triclinic;

  // Scatter views to use when calculating the per-molecule arrays across multiple threads
  Kokkos::Experimental::ScatterView<double*, typename AT::t_float_1d::array_layout> dup_massproc;
  Kokkos::Experimental::ScatterView<X_FLOAT*[3], typename AT::t_x_array::array_layout> dup_comproc;
  Kokkos::Experimental::ScatterView<X_FLOAT*[3], typename AT::t_v_array::array_layout> dup_vcmproc;
  Kokkos::Experimental::ScatterView<double*, typename AT::t_float_1d::array_layout> dup_keproc;

};

}    // namespace LAMMPS_NS

#endif
#endif
