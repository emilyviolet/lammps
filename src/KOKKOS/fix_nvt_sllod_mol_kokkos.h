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
FixStyle(nvt/sllod/mol/kk,FixNVTSllodMolKokkos<LMPDeviceType>);
FixStyle(nvt/m-sllod/mol/kk,FixNVTSllodMolKokkos<LMPDeviceType>);
FixStyle(nvt/sllod/mol/kk/device,FixNVTSllodMolKokkos<LMPDeviceType>);
FixStyle(nvt/m-sllod/mol/kk/device,FixNVTSllodMolKokkos<LMPDeviceType>);
FixStyle(nvt/sllod/mol/kk/host,FixNVTSllodMolKokkos<LMPHostType>);
FixStyle(nvt/m-sllod/mol/kk/host,FixNVTSllodMolKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_NVT_SLLOD_MOL_KOKKOS_H
#define LMP_FIX_NVT_SLLOD_MOL_KOKKOS_H

#include "fix_nh_kokkos.h"
#include "kokkos_few.h"
#include "kokkos_type.h"
#include "fix_property_mol_kokkos.h"

namespace LAMMPS_NS {

struct TagFixNVTSllodMol_vtemp{};
struct TagFixNVTSllodMol_x1{};
struct TagFixNVTSllodMol_x2{};

template<class DeviceType>
class FixNVTSllodMolKokkos : public FixNHKokkos<DeviceType>  {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixNVTSllodMolKokkos(class LAMMPS *, int, char **);
  ~FixNVTSllodMolKokkos();
  void post_constructor() override;

  void init() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVTSllodMol_vtemp, const int &i) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVTSllodMol_x1, const int &i) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVTSllodMol_x2, const int &i) const;


 protected:
  int molpropflag;    // 1 = molprop created by nvt/sllod/mol, 0 = user supplied
  char *id_molpropKK;   // Name of property/molecule fix
  FixPropertyMolKokkos<DeviceType> *molpropKK;

 private:
  void nh_v_temp() override;
  void nve_x() override;

 protected:
  // Device views
  // Per-atom quantities
  typename AT::t_x_array x;             // Atomic coords
  typename AT::t_v_array v;             // Atomic velocities
  typename AT::t_tagint_1d molID;
  typename AT::t_int_1d type;
  typename AT::t_int_1d mask;
  typename AT::t_imageint_1d image;
  typename AT::t_float_1d rmass;
  typename AT::t_float_1d mass;

  // Per-molecule quantities
  typename AT::t_v_array vcm;           // per molecule center of mass velocity
  typename AT::t_v_array com;           // per molecule center of mass velocity
  typename AT::t_v_array vdelu;
  typename AT::t_f_array_const f;


  // Small, temporary device side arrays
  Few<double, 6> d_grad_u;
  Few<double, 3> d_vfac;
  Few<double, 3> d_xfac;
  double dt4, dtv2;

  // Variables for domain unwrapping
  Few<double, 6> h;
  Few<double, 3> prd;
  int triclinic;

  class DomainKokkos *domainKK;
  class AtomKokkos *atomKK;

};

}    // namespace LAMMPS_NS

#endif
#endif
