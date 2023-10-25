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
ComputeStyle(temp/mol/kk,ComputeTempMolKokkos<LMPDeviceType>);
ComputeStyle(temp/mol/kk/device,ComputeTempMolKokkos<LMPDeviceType>);
ComputeStyle(temp/mol/kk/host,ComputeTempMolKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_TEMP_MOL_KOKKOS_H
#define LMP_COMPUTE_TEMP_MOL_KOKKOS_H

#include "kokkos_type.h"
#include "compute_temp_mol.h"
#include "fix_property_mol_kokkos.h"

namespace LAMMPS_NS {

struct TagComputeTempMolScalar{};
struct TagComputeTempMolVector{};

template<class DeviceType>
class ComputeTempMolKokkos : public ComputeTempMol {
 public:
  // Struct type for vector temperature components and Kokkos reduction
  struct s_CTEMP {
    // t0 = scalar/diagonal temperature
    // t[1-5] = off-diagonal
    double t0, t1, t2, t3, t4, t5;
    KOKKOS_INLINE_FUNCTION
    s_CTEMP() {
      t0 = t1 = t2 = t3 = t4 = t5 = 0.0;
    }
    KOKKOS_INLINE_FUNCTION
    s_CTEMP& operator+=(const s_CTEMP &rhs) {
      t0 += rhs.t0;
      t1 += rhs.t1;
      t2 += rhs.t2;
      t3 += rhs.t3;
      t4 += rhs.t4;
      t5 += rhs.t5;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile s_CTEMP &rhs) volatile {
      t0 += rhs.t0;
      t1 += rhs.t1;
      t2 += rhs.t2;
      t3 += rhs.t3;
      t4 += rhs.t4;
      t5 += rhs.t5;
    }
  };

  typedef s_CTEMP CTEMP;
  typedef CTEMP value_type;
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  // Class methods
  ComputeTempMolKokkos(class LAMMPS *, int, char **);
  ~ComputeTempMolKokkos() override;

  void init() override;
  void setup() override;
  double compute_scalar() override;
  void compute_vector() override;
  void dof_compute();

  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeTempMolScalar, const int&, CTEMP&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeTempMolVector, const int&, CTEMP&) const;
 protected:
  char *id_molpropKK;
  FixPropertyMolKokkos<DeviceType>* molpropKK;

 private:
  int nmax;
  double adof, cdof, tfactor;

  void allocate();
 
  // Device views
  // Per-molecule types
  typename AT::t_float_1d molmass;       // per molecule mass view
  typename AT::t_x_array com;           // per molecule center of mass view in unwrapped coords
  typename AT::t_v_array vcm;           // per molecule center of mass velocity

  // Per-atom types
  typename AT::t_x_array_randomread x;
  typename AT::t_v_array v;
  typename AT::t_v_array vbiasall;
  typename AT::t_float_1d_randomread rmass;
  typename AT::t_float_1d_randomread mass;
  typename AT::t_int_1d_randomread type;
  typename AT::t_int_1d_randomread mask;
  typename AT::t_int_1d molID;


  private:
    class DomainKokkos *domainKK;
  };

}    // namespace LAMMPS_NS

#endif
#endif
