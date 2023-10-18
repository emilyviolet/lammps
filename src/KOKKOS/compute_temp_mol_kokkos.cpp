// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Emily Kahl, Stephen Sanderson, Shern Tee (Uni of QLD)
------------------------------------------------------------------------- */

#include "compute_temp_mol_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain_kokkos.h"
#include "error.h"
#include "fix.h"
#include "fix_property_mol_kokkos.h"
#include "force.h"
#include "group.h"
#include "impl/Kokkos_Combined_Reducer.hpp"
#include "memory_kokkos.h"
#include "modify.h"
#include "pointers.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputeTempMolKokkos<DeviceType>::ComputeTempMolKokkos(LAMMPS *lmp, int narg, char **arg) :
  ComputeTempMol(lmp, narg, arg), molpropKK(nullptr),
  id_molpropKK(nullptr)
{
  kokkosable = 1;
  if (narg != 4) error->all(FLERR,"Illegal compute temp/mol command");

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 1;  // TODO(SS): if thermo_modify norm yes is set, then the vector will be divided by the number of atoms, which is incorrect.
  tempflag = 1;
  tempbias = 0;
  maxbias = 0;
  array_flag = 0;

  domainKK = (DomainKokkos *) domain;
  atomKK = (AtomKokkos *) atom;
  adof = domainKK->dimension;
  cdof = 0.0;

  // vector data
  vector = new double[size_vector];

  // per-atom allocation
  nmax = 0;

  id_molpropKK = utils::strdup(arg[3]);
}

template<class DeviceType>
ComputeTempMolKokkos<DeviceType>::~ComputeTempMolKokkos(){
  if (copymode) return;
  delete [] id_molpropKK;
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMolKokkos<DeviceType>::init()
{
  // Get id of molpropKK
  molpropKK = dynamic_cast<FixPropertyMolKokkos<DeviceType> *>(modify->get_fix_by_id(id_molpropKK));
  if (molpropKK == nullptr)
    error->all(FLERR, "Compute temp/mol could not find a fix property/mol with id {}", id_molpropKK);
  // if (!molpropKK->mass_flag)
  //   error->all(FLERR, "Compute temp/mol requires fix property/mol with the mass or com flag");
  if (igroup != molpropKK->igroup)
    error->all(FLERR, "Fix property/mol must be defined for the same group as compute temp/mol");

  molpropKK->request_mass();
  molpropKK->request_vcm(); 
}

template<class DeviceType>
void ComputeTempMolKokkos<DeviceType>::setup()
{
  dynamic = 0;
  if (dynamic_user || group->dynamic[igroup]) dynamic = 1;
  dof_compute();
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
double ComputeTempMolKokkos<DeviceType>::compute_scalar()
{
  int i;
  invoked_scalar = update->ntimestep;

  tagint molmax = molpropKK->molmax;

  molmass = molpropKK->k_mass.template view<DeviceType>();//.template view<DeviceType()>;
  vcm = molpropKK->k_vcm.template view<device_type>();
  double* ke_singles = molpropKK->ke_singles;

  v = atomKK->k_v.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  tagint m;

  molpropKK->vcm_compute();
  // Tally up the molecule COM velocities to get the kinetic temperature
  // ke_singles first, since those are always a fixed-size
  double t = ke_singles[0]+ke_singles[1]+ke_singles[2];
  CTEMP t_kk;

  // No need for MPI reductions, since every processor knows the molecule VCMs
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagComputeTempMolScalar>(0, molmax), *this, t_kk);
  copymode = 0;
  t += t_kk.t0;

  // final temperature
  if (dynamic)
    dof_compute();
  if (dof < 0.0)
    error->all(FLERR,"Temperature compute degrees of freedom < 0");
  scalar = t*tfactor;
  return scalar;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void ComputeTempMolKokkos<DeviceType>::operator()(TagComputeTempMolScalar, const int &m, CTEMP &t_kk) const
{
    t_kk.t0 += (vcm(m,0)*vcm(m,0) + vcm(m,1)*vcm(m,1) + vcm(m,2)*vcm(m,2)) *
          molmass(m);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMolKokkos<DeviceType>::compute_vector()
{
  int i;

  invoked_vector = update->ntimestep;

  tagint molmax = molpropKK->molmax;
  molmass = molpropKK->k_mass.template view<DeviceType>();
  vcm = molpropKK->k_vcm.template view<DeviceType>();
  double* ke_singles = molpropKK->ke_singles;

  v = atomKK->k_v.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  int nlocal = atom->nlocal;
  tagint m;

  double massone,t[6];
  for (i = 0; i < 6; i++) t[i] = 0.0;

  molpropKK->vcm_compute();
  // Tally up the molecule COM velocities to get the kinetic temperature
  // No need for MPI reductions, since every processor knows the molecule VCMs
  CTEMP t_kk;
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagComputeTempMolVector>(0, molmax), *this, t_kk);
  copymode = 0;

  t[0] = t_kk.t0;
  t[1] = t_kk.t1;
  t[2] = t_kk.t2;
  t[3] = t_kk.t3;
  t[4] = t_kk.t4;
  t[5] = t_kk.t5;

  // final KE. Include contribution from single atoms if there are any
  for (i = 0; i < 6; i++) vector[i] = (t[i]+ke_singles[i])*force->mvv2e;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void ComputeTempMolKokkos<DeviceType>::operator()(TagComputeTempMolVector, const int &m, CTEMP &t_kk) const
{
      t_kk.t0 += molmass[m] * vcm(m,0) * vcm(m, 0);
      t_kk.t1 += molmass[m] * vcm(m,1) * vcm(m, 1);
      t_kk.t2 += molmass[m] * vcm(m,2) * vcm(m, 2);
      t_kk.t3 += molmass[m] * vcm(m,0) * vcm(m, 1);
      t_kk.t4 += molmass[m] * vcm(m,0) * vcm(m, 2);
      t_kk.t5 += molmass[m] * vcm(m,1) * vcm(m, 2);
}
/* ----------------------------------------------------------------------
   Degrees of freedom for molecular temperature
------------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMolKokkos<DeviceType>::dof_compute()
{
  // TODO(SS): fix_dof will be incorrect for rigid molecules, since we only care
  //           about CoM momentum. Ignoring it for now, but maybe look
  //           into calculating the number of intermolecular constraints which
  //           should be counted.
  adjust_dof_fix();
  if (fix_dof != 0 && comm->me == 0)
    error->warning(FLERR,"Ignoring dof constraints due to fixes in compute "
        "temp/mol. These must be accounted for manually since intramolecular "
        "constraints should be ignored.");

  // Count atoms in the group that aren't part of a molecule
  //int *mask = atom->mask;
  mask = atomKK->k_mask.view<DeviceType>();
  //todo - check if correct
  int nlocal = atom->nlocal;
  bigint nsingle_local = 0, nsingle;
  Kokkos::parallel_reduce(nlocal, LAMMPS_LAMBDA(int i, bigint& nsingle_local) {
    if (mask[i] & groupbit && atom->molecule[i] == 0)
      nsingle_local += 1;
  }, nsingle_local);

  MPI_Allreduce(&nsingle_local,&nsingle,1,MPI_LMP_BIGINT,MPI_SUM,world);

  // Make sure molecule count is up to date
  if (molpropKK->dynamic_group || molpropKK->dynamic_mols) {
      if (molpropKK->count_step != update->ntimestep) {
        if (molpropKK->mass_step != update->ntimestep)
          molpropKK->mass_compute();
        molpropKK->count_molecules();
      }
  }

  // Calculate dof from number of molecules with at least 1 atom in the group
  dof = domain->dimension * (molpropKK->nmolecule + nsingle);

  dof -= extra_dof; // + fix_dof;
  if (dof > 0)
    tfactor = force->mvv2e / (dof * force->boltz);
  else
    tfactor = 0.0;
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */
//template <class DeviceType>
//double ComputeTempMolKokkos<DeviceType>::memory_usage()
//{
//  double bytes = 0;
//  if (molpropKK != nullptr)
//    bytes += (bigint) molpropKK->molmax * 6 * sizeof(double);
//
//  return bytes;
//}

namespace LAMMPS_NS {
template class ComputeTempMolKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeTempMolKokkos<LMPHostType>;
#endif
}

