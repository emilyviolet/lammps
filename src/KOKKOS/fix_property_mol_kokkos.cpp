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

#include "fix_property_mol_kokkos.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;


/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixPropertyMolKokkos<DeviceType>::FixPropertyMolKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixPropertyMol(lmp, narg, arg)
{
  kokkosable = 1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixPropertyMolKokkos<DeviceType>::~FixPropertyMolKokkos()
{
  if (mass_flag)
  {
    memoryKK->destroy_kokkos(k_mass, mass);
    memoryKK->destroy_kokkos(k_massproc, massproc);
  }
  if(com_flag)
  {
    memoryKK->destroy_kokkos(k_com, com);
    memoryKK->destroy_kokkos(k_comproc, comproc);
  }
  if(vcm_flag)
  {
    memoryKK->destroy_kokkos(k_vcm, vcm);
    memoryKK->destroy_kokkos(k_vcmproc, vcmproc);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixPropertyMolKokkos<DeviceType>::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  return mask;
}

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::request_com() {
  if (com_flag) return;
  com_flag = 1;
  request_mass();
  memoryKK->create_kokkos(k_com, com, molmax, "property/mol:com");
  memoryKK->create_kokkos(k_comproc, comproc, molmax, "property/mol:comproc");
}

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::request_vcm() {
  if (vcm_flag) return;
  vcm_flag = 1;
  request_mass();
  memoryKK->create_kokkos(k_vcm, vcm, molmax, "property/mol:vcm");
  memoryKK->create_kokkos(k_vcmproc, vcmproc, molmax, "property/mol:vcmproc");
}
template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::request_mass() {
  if (mass_flag) return;
  mass_flag = 1;
  memoryKK->create_kokkos(k_mass, mass, molmax, "property/mol:mass");
  memoryKK->create_kokkos(k_massproc, massproc, molmax, "property/mol:massproc");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::init()
{
  // Error if system doesn't track molecule ids.
  // Check here since atom_style could change before run.

  if (!atom->molecule_flag)
    error->all(FLERR,
        "Fix property/mol when atom_style does not define a molecule attribute");
}

/* ----------------------------------------------------------------------
   Need to calculate mass and CoM before main setup() calls since those
   could rely on the memory being allocated (e.g. for virial tallying)
---------------------------------------------------------------------- */
template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::setup_pre_force(int /*vflag*/) {
  dynamic_group = group->dynamic[igroup];
  grow_permolecule();

  // com_compute also computes mass if dynamic_group is set
  // so no need to call mass_compute in that case
  if (mass_flag && !dynamic_group) mass_compute();
  if (com_flag) com_compute();
  if (mass_flag) count_molecules();
}

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::setup_pre_force_respa(int vflag, int ilevel) {
  if (ilevel == 0) setup_pre_force(vflag);
}


/* ----------------------------------------------------------------------
   Calculate number of molecules and grow permolecule arrays if needed.
   Grows to the maximum of previous max. mol id + grow_by and new max. mol id
   if either is larger than nmax.
   Returns true if the number of molecules (max. mol id) changed.
------------------------------------------------------------------------- */

template<class DeviceType>
bool FixPropertyMolKokkos<DeviceType>::grow_permolecule(int grow_by) {
  // Calculate maximum molecule id
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  tagint maxone = -1;
  for (int i = 0; i < nlocal; i++)
    if (molecule[i] > maxone) maxone = molecule[i];
  tagint maxall;
  MPI_Allreduce(&maxone, &maxall, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  if (maxall > MAXSMALLINT)
    error->all(FLERR, "Molecule IDs too large for fix property/mol");

  tagint old_molmax = molmax;
  tagint new_size = molmax + grow_by;
  molmax = maxall;
  new_size = MAX(molmax, new_size);

  // Grow arrays as needed
  if (nmax < new_size) {
    nmax = new_size;
    if (mass_flag)
    {
      memoryKK->grow_kokkos(k_mass, mass, nmax, "property/mol:mass");
      memoryKK->grow_kokkos(k_massproc, massproc, nmax, "property/mol:massproc");
    }
    if (com_flag)
    {
      memoryKK->grow_kokkos(k_com, com, nmax, "property/mol:com");
      memoryKK->grow_kokkos(k_comproc, comproc, nmax, "property/mol:comproc");
    }
    if (vcm_flag)
    {
      memoryKK->grow_kokkos(k_vcm, vcm, nmax, "property/mol:vcm");
      memoryKK->grow_kokkos(k_vcmproc, vcmproc, nmax, "property/mol:vcmproc");
    }
  }

  size_array_rows = static_cast<int>(molmax);
  return old_molmax != molmax;
}


/* ----------------------------------------------------------------------
   Count the number of molecules with non-zero mass.
   Mass of molecules is only counted from atoms in the group, so count is
   the number of molecules in the group.
------------------------------------------------------------------------- */
template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::count_molecules() {
  count_step = update->ntimestep;
  nmolecule = 0;
  for (tagint m = 0; m < molmax; ++m)
    if (mass[m] > 0.0) ++nmolecule;
}

/* ----------------------------------------------------------------------
   Update total mass of each molecule
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::mass_compute() {
  
  // Kokkos stuff

  mass_step = update->ntimestep;
  if (dynamic_mols) grow_permolecule();
  if (molmax == 0) return;
  double massone;
  for (tagint m = 0; m < molmax; ++m)
    massproc[m] = 0.0;

  for (int i = 0; i < atom->nlocal; ++i) {
    if (groupbit & atom->mask[i]) {
      tagint m = atom->molecule[i]-1;
      if (m < 0) continue;
      if (atom->rmass) massone = atom->rmass[i];
      else massone = atom->mass[atom->type[i]];
      massproc[m] += massone;
    }
  }
  MPI_Allreduce(massproc,mass,molmax,MPI_DOUBLE,MPI_SUM,world);
}

/* ----------------------------------------------------------------------
   Calculate center of mass of each molecule in unwrapped coords
   Also update molecular mass if group is dynamic
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::com_compute() {
  com_step = update->ntimestep;
  // Recalculate mass if number of molecules (max. mol id) changed, or if
  // group is dynamic
  bool recalc_mass = dynamic_group;
  if (dynamic_mols) recalc_mass |= grow_permolecule();
  if (molmax == 0) return;

  int nlocal = atom->nlocal;
  tagint *molecule = atom->molecule;

  int *type = atom->type;
  double *amass = atom->mass;
  double *rmass = atom->rmass;
  double **x = atom->x;
  double **v = atom->v;
  double massone, unwrap[3];

  for (int m = 0; m < molmax; ++m) {
    comproc[m][0] = 0.0;
    comproc[m][1] = 0.0;
    comproc[m][2] = 0.0;
  }

  if (recalc_mass) {
    mass_step = update->ntimestep;
    for (tagint m = 0; m < molmax; ++m)
      massproc[m] = 0.0;
  }

  for (int i = 0; i < nlocal; ++i) {
    if (groupbit & atom->mask[i]) {
      tagint m = molecule[i]-1;
      if (m < 0) continue;
      if (rmass) massone = rmass[i];
      else massone = amass[type[i]];

      // NOTE: if FP error becomes a problem here in long-running
      //       simulations, could maybe do something clever with
      //       image flags to reduce it, but MPI makes that difficult,
      //       and it would mean needing to store image flags for CoM
      domain->unmap(x[i],atom->image[i],unwrap);
      comproc[m][0] += unwrap[0] * massone;
      comproc[m][1] += unwrap[1] * massone;
      comproc[m][2] += unwrap[2] * massone;
      if (recalc_mass) massproc[m] += massone;
    }
  }

  MPI_Allreduce(&comproc[0][0],&com[0][0],3*molmax,MPI_DOUBLE,MPI_SUM,world);
  if (recalc_mass) MPI_Allreduce(massproc,mass,molmax,MPI_DOUBLE,MPI_SUM,world);

  for (int m = 0; m < molmax; ++m) {
    // Some molecule ids could be skipped (not assigned atoms)
    if (mass[m] > 0.0) {
      com[m][0] /= mass[m];
      com[m][1] /= mass[m];
      com[m][2] /= mass[m];
    } else {
      com[m][0] = com[m][1] = com[m][2] = 0.0;
    }
  }
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

template<class DeviceType>
double FixPropertyMolKokkos<DeviceType>::memory_usage()
{
  double bytes = 0.0;
  if (mass_flag) bytes += nmax * 2 * sizeof(double);
  if (com_flag)  bytes += nmax * 6 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   basic array output
------------------------------------------------------------------------- */

template<class DeviceType>
double FixPropertyMolKokkos<DeviceType>::compute_array(int imol, int col)
{
  if (imol > static_cast<int>(molmax))
    error->all(FLERR, fmt::format(
      "Cannot request info for molecule {} from fix property/mol (molmax = {})",
      imol, molmax));

  if (col == 3) {
    // Mass requested
    if (!mass_flag)
      error->all(FLERR, "This fix property/mol does not calculate mass");
    if (dynamic_group && mass_step != update->ntimestep) mass_compute();
    return mass[imol];
  } else {
    // CoM requested
    if (!com_flag)
      error->all(FLERR, "This fix property/mol does not calculate CoM");
    if (com_step != update->ntimestep) com_compute();
    return com[imol][col];
  }
}

namespace LAMMPS_NS {
template class FixPropertyMolKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixPropertyMolKokkos<LMPHostType>;
#endif
}

