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

#include "Kokkos_Core_fwd.hpp"
#include "atom.h"
#include "atom_kokkos.h"
#include "domain.h"
#include "domain_kokkos.h"
#include "error.h"
#include "group.h"
#include "kokkos_type.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "update.h"
#include "kokkos_few.h"
#include "kokkos.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;


/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixPropertyMolKokkos<DeviceType>::FixPropertyMolKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixPropertyMol(lmp, narg, arg)
{
  kokkosable = 1;
  // Now check if any of the mass, COM or VCM flags have been requested by the base class's constructor
  // and call the relevant memory allocation functions. Set to zero first so the Kokkos registration
  // functions actually do stuff
  if(mass_flag)
  {
    mass_flag = 0;
    request_mass();
  }
  if(com_flag)
  {
    com_flag = 0;
    request_com();
  }
  if(vcm_flag)
  {
    vcm_flag = 0;
    request_vcm();
  }
  
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixPropertyMolKokkos<DeviceType>::~FixPropertyMolKokkos()
{
  if (copymode) return;
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
    memoryKK->destroy_kokkos(k_ke_singles, ke_singles);
    memoryKK->destroy_kokkos(k_keproc, keproc);
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
  memoryKK->create_kokkos(k_ke_singles, ke_singles, 6, "property/mol:ke_singles");
  memoryKK->create_kokkos(k_keproc, keproc, 6, "property/mol:keproc");
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
  if (vcm_flag) vcm_compute();
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
  k_mass.sync<DeviceType>();
  d_mass = k_mass.view<DeviceType>();

  for(int m = 0; m < molmax; m++) {
    //printf("Host: m = %d, mass = %lg\n", m, k_mass.h_view[m]);
  }

  {
    // Local copy of mass for LAMBDA capture
    auto l_mass = this->d_mass;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,molmax), LAMMPS_LAMBDA (const int m, int &rcount) {
      if (l_mass[m] > 0.0)
        rcount += 1;
      //printf("Kernel: m = %d, mass = %lg\n", m, l_mass[m]);
    }, nmolecule);
  }
}


/* ----------------------------------------------------------------------
   Update total mass of each molecule
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::mass_compute() {
  
  atomKK->sync(execution_space,datamask_read);

  atom_type = atomKK->k_type.view<DeviceType>();
  atom_mask = atomKK->k_mask.view<DeviceType>();
  atom_molID = atomKK->k_molecule.view<DeviceType>();
  // Only instantiate the necessary mass view
  if (atomKK->rmass) {
    atomKK->k_rmass.sync<DeviceType>();
    atom_rmass = atomKK->k_rmass.view<DeviceType>();
  } else {
    atomKK->k_mass.sync<DeviceType>();
    atom_mass = atomKK->k_mass.view<DeviceType>();
  }

  // The device-side of k_mass gets marked as modified by memoryKK->grow_kokkos, so Kokkos will complain
  // if we mark the host-side as modified after the MPI call (technically shouldn't mark both sides as
  // modified at the same time). We're going to be over-writing the existing values during this
  // function, though, so it should be safe to just tell Kokkos to ignore the current sync status
  k_mass.clear_sync_state();

  d_massproc = k_massproc.view<DeviceType>();

  tagint nlocal = atomKK->nlocal;

  mass_step = update->ntimestep;
  // Grow arrays if the molecules might have changed this step
  if (dynamic_mols) grow_permolecule();
  if (molmax == 0) return;

  // Zero arrays
  Kokkos::deep_copy(d_massproc, 0.0);

  // Now set up the scatter view to track massproc so we can reduce into the elements without data races
  scatter_massproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum>(d_massproc);
  copymode = 1;
  if (atomKK->rmass) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_mass_compute<1> >(0, nlocal), *this);
  } else {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_mass_compute<0> >(0, nlocal), *this);
  }
  copymode = 0;
  Kokkos::Experimental::contribute(d_massproc, scatter_massproc);

  // TODO EVK: Should really have a code-path to keep this all on the device if LAMMPS is compiled with GPU-aware MPI
  k_massproc.template modify<DeviceType>();
  k_massproc.template sync<LMPHostType>();
  MPI_Allreduce(k_massproc.h_view.data(), k_mass.h_view.data(), molmax, MPI_DOUBLE, MPI_SUM, world);
  k_mass.modify<LMPHostType>();
  k_mass.sync<DeviceType>();
}

template<class DeviceType>
template<int RMASS>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_mass_compute<RMASS>, const int&i) const
{
  auto a_massproc = scatter_massproc.template access();
  if (RMASS) {
    if (groupbit & atom_mask[i]) {
      tagint m = atom_molID[i]-1;
      if (m < 0) return;
      a_massproc[m] += atom_rmass[i];
    }
  } else {
    if (groupbit & atom_mask[i]) {
      tagint m = atom_molID[i]-1;
      if (m < 0) return;
      a_massproc[m] += atom_mass[atom_type[i]];
    } 
  }
}

/* ----------------------------------------------------------------------
   Calculate center of mass of each molecule in unwrapped coords
   Also update molecular mass if group is dynamic
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::com_compute() {
  atomKK->sync(execution_space,datamask_read);
  com_step = update->ntimestep;
  // Recalculate mass if number of molecules (max. mol id) changed, or if
  // group is dynamic
  bool recalc_mass = dynamic_group;
  if (dynamic_mols) recalc_mass |= grow_permolecule();
  if (molmax == 0) return;
  if (recalc_mass) {
    mass_compute();
  }

  int nlocal = atomKK->nlocal;
  atom_type = atomKK->k_type.view<DeviceType>();
  atom_mask = atomKK->k_mask.view<DeviceType>();
  atom_molID = atomKK->k_molecule.view<DeviceType>();
  atom_x = atomKK->k_x.view<DeviceType>();
  atom_image = atomKK->k_image.view<DeviceType>();
  // Only instantiate the necessary mass view
  if (atomKK->rmass) {
    atomKK->k_rmass.sync<DeviceType>();
    atom_rmass = atomKK->k_rmass.view<DeviceType>();
  } else {
    atomKK->k_mass.sync<DeviceType>();
    atom_mass = atomKK->k_mass.view<DeviceType>();
  }

  // The device-side of k_com and friends gets marked as modified by memoryKK->grow_kokkos, so Kokkos will complain
  // if we mark the host-side as modified after the MPI call (technically shouldn't mark both sides as
  // modified at the same time). We're going to be over-writing the existing values during this
  // function, though, so it should be safe to just tell Kokkos to ignore the current sync status
  k_com.clear_sync_state();

  d_com = k_com.view<DeviceType>();
  d_comproc = k_comproc.view<DeviceType>();
  d_mass = k_mass.view<DeviceType>();

  // Zero the array
  Kokkos::deep_copy(d_comproc, 0.0);

  // Get the box properties from domainKK so we can remap inside the kernel
  prd = Few<double, 3>(domain->prd);
  h = Few<double, 6>(domain->h);
  triclinic = domain->triclinic;

  // Now we need a scatter view to reduce into. Let Kokkos decide whether to duplicate or not
  scatter_comproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum>(d_comproc);
  copymode = 1;
  if (atomKK->rmass) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_com_compute<1> >(0, nlocal), *this);
  } else {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_com_compute<0> >(0, nlocal), *this);
  }
  copymode = 0;

  Kokkos::Experimental::contribute(d_comproc, scatter_comproc);

  k_comproc.template modify<DeviceType>();
  k_comproc.template sync<LMPHostType>();
  MPI_Allreduce(k_comproc.h_view.data(),k_com.h_view.data(),3*molmax,MPI_DOUBLE,MPI_SUM,world);
  k_com.modify<LMPHostType>();
  k_com.sync<DeviceType>();

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_com_scale>(0, molmax), *this);
  copymode = 0;
  k_com.modify<DeviceType>();
  k_com.sync<LMPHostType>();
}

template<class DeviceType>
template<int RMASS>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_com_compute<RMASS>, const int& i) const
{
  auto a_comproc = scatter_comproc.template access();
  if(groupbit & atom_mask[i])
  {
    // TODO EVK: Refactor, too much copy-paste
    if(RMASS){
      tagint m = atom_molID[i]-1;
      if(m < 0) return;
      double massone = atom_rmass[i];
      // Need to unwrap the coords. Make a Kokkos Few first to interface with domainKK
      // TODO EVK: Also, this probably needs to get hoisted outside the kernel so we're not
      // constantly allocating memory
      Few<double, 3> x_i;
      x_i[0] = atom_x(i, 0);
      x_i[1] = atom_x(i, 1);
      x_i[2] = atom_x(i, 2);
      auto unwrap = DomainKokkos::unmap(prd, h, triclinic, x_i, atom_image[i]);
      a_comproc(m, 0) += unwrap[0] * massone;
      a_comproc(m, 1) += unwrap[1] * massone;
      a_comproc(m, 2) += unwrap[2] * massone;
    } else {
      tagint m = atom_molID[i]-1;
      if(m < 0) return;
      double massone = atom_mass[atom_type[i]];
      // Need to unwrap the coords. Make a Kokkos Few first to interface with domainKK
      Few<double, 3> x_i;
      x_i[0] = atom_x(i, 0);
      x_i[1] = atom_x(i, 1);
      x_i[2] = atom_x(i, 2);
      auto unwrap = DomainKokkos::unmap(prd, h, triclinic, x_i, atom_image[i]);
      a_comproc(m, 0) += unwrap[0] * massone;
      a_comproc(m, 1) += unwrap[1] * massone;
      a_comproc(m, 2) += unwrap[2] * massone;
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_com_scale, const int &m) const
{
  if (d_mass[m] > 0.0) {
    d_com(m, 0) /= d_mass[m];
    d_com(m, 1) /= d_mass[m];
    d_com(m, 2) /= d_mass[m];
  } else {
    d_com(m, 0) = 0.0; 
    d_com(m, 1) = 0.0;
    d_com(m, 2) = 0.0;
  }
}

/* ----------------------------------------------------------------------
   Calculate center of mass velocity of each molecule 
   Also update molecular mass if group is dynamic
------------------------------------------------------------------------- */
template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::vcm_compute()
{
  atomKK->sync(execution_space,datamask_read);
  vcm_step = update->ntimestep;
  // Recalculate mass if number of molecules (max. mol id) changed, or if
  // group is dynamic
  bool recalc_mass = dynamic_group;
  if (dynamic_mols) recalc_mass |= grow_permolecule();
  if (molmax == 0) return;
  if (recalc_mass) {
    mass_compute();
  }

  int nlocal = atomKK->nlocal;
  // Kokkos variables
  atom_molID = atomKK->k_molecule.view<DeviceType>();
  atom_type = atomKK->k_type.view<DeviceType>();
  atom_mask = atomKK->k_mask.view<DeviceType>();
  atom_v = atomKK->k_v.view<DeviceType>();
  atom_image = atomKK->k_image.view<DeviceType>();
 
  // Only instantiate the necessary mass view
  if (atomKK->rmass) {
    atomKK->k_rmass.sync<DeviceType>();
    atom_rmass = atomKK->k_rmass.view<DeviceType>();
  } else {
    atomKK->k_mass.sync<DeviceType>();
    atom_mass = atomKK->k_mass.view<DeviceType>();
  }

  // The device-side of k_vcm and friends gets marked as modified by memoryKK->grow_kokkos, so Kokkos will complain
  // if we mark the host-side as modified after the MPI call (technically shouldn't mark both sides as
  // modified at the same time). We're going to be over-writing the existing values during this
  // function, though, so it should be safe to just tell Kokkos to ignore the current sync status
  k_vcm.clear_sync_state();
  k_ke_singles.clear_sync_state();

  d_vcm = k_vcm.view<DeviceType>();
  d_vcmproc = k_vcmproc.view<DeviceType>();
  d_keproc = k_keproc.view<DeviceType>();

  // Zero the arrays
  Kokkos::deep_copy(d_vcmproc, 0.0);
  Kokkos::deep_copy(d_keproc, 0.0);

  // Scatter view to reduce into in VCM functor
  scatter_vcmproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum>(d_vcmproc);
  scatter_keproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum>(d_keproc);
  copymode = 1;
  if (atomKK->rmass) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcm_compute<1> >(0, nlocal), *this);
  } else {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcm_compute<0> >(0, nlocal), *this);
  }

  Kokkos::Experimental::contribute(d_vcmproc, scatter_vcmproc);
  Kokkos::Experimental::contribute(d_keproc, scatter_keproc);

  k_vcmproc.template modify<DeviceType>();
  k_vcmproc.template sync<LMPHostType>();
  k_keproc.template modify<DeviceType>();
  k_keproc.template sync<LMPHostType>();
  MPI_Allreduce(k_vcmproc.h_view.data(),k_vcm.h_view.data(),3*molmax,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(k_keproc.h_view.data(),k_ke_singles.h_view.data(), 6, MPI_DOUBLE, MPI_SUM, world);
  k_vcm.modify<LMPHostType>();
  k_vcm.sync<DeviceType>();
  k_ke_singles.modify<LMPHostType>();
  k_ke_singles.sync<DeviceType>();

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcm_scale>(0, molmax), *this);
  copymode = 0;
  k_vcm.template modify<DeviceType>();
  k_vcm.template sync<LMPHostType>();
}

template<class DeviceType>
template<int RMASS>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_vcm_compute<RMASS>, const int& i) const
{
  auto a_vcmproc = scatter_vcmproc.access();
  auto a_keproc = scatter_keproc.access();
  if(RMASS){
    if (groupbit & atom_mask[i]) {
      double massone = atom_rmass[i];
      tagint m = atom_molID[i]-1;
      if (m < 0) {
        a_keproc[0] += atom_v(i, 0)*atom_v(i, 0)*massone;
        a_keproc[1] += atom_v(i, 1)*atom_v(i, 1)*massone;
        a_keproc[2] += atom_v(i, 2)*atom_v(i, 2)*massone;
        a_keproc[3] += atom_v(i, 0)*atom_v(i, 1)*massone;
        a_keproc[4] += atom_v(i, 0)*atom_v(i, 2)*massone;
        a_keproc[5] += atom_v(i, 1)*atom_v(i, 2)*massone;
      } else {
        a_vcmproc(m, 0) += atom_v(i, 0) * massone;
        a_vcmproc(m, 1) += atom_v(i, 1) * massone;
        a_vcmproc(m, 2) += atom_v(i, 2) * massone;
      }
    }
  } else {
    if (groupbit & atom_mask[i]) {
      double massone = atom_mass[atom_type[i]];
      tagint m = atom_molID[i]-1;
      if (m < 0) {
        a_keproc[0] += atom_v(i, 0)*atom_v(i, 0)*massone;
        a_keproc[1] += atom_v(i, 1)*atom_v(i, 1)*massone;
        a_keproc[2] += atom_v(i, 2)*atom_v(i, 2)*massone;
        a_keproc[3] += atom_v(i, 0)*atom_v(i, 1)*massone;
        a_keproc[4] += atom_v(i, 0)*atom_v(i, 2)*massone;
        a_keproc[5] += atom_v(i, 1)*atom_v(i, 2)*massone;
      } else {
        a_vcmproc(m, 0) += atom_v(i, 0) * massone;
        a_vcmproc(m, 1) += atom_v(i, 1) * massone;
        a_vcmproc(m, 2) += atom_v(i, 2) * massone;
      }
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_vcm_scale, const int& m) const
{
  if(d_mass[m] > 0.0)
  {
    d_vcm(m, 0) /= d_mass[m];
    d_vcm(m, 1) /= d_mass[m];
    d_vcm(m, 2) /= d_mass[m];
  } else {
    d_vcm(m, 0) = 0.0;
    d_vcm(m, 1) = 0.0;
    d_vcm(m, 2) = 0.0;
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
  if (vcm_flag)  bytes += nmax * 6 * sizeof(double);
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

