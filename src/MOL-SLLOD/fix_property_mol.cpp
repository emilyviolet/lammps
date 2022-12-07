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

#include "fix_property_mol.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;


/* ---------------------------------------------------------------------- */

FixPropertyMol::FixPropertyMol(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), mass(nullptr), com(nullptr), massproc(nullptr),
    comproc(nullptr)
{
  if (narg < 3) error->all(FLERR, "Illegal fix property/mol command");

  int iarg = 3;

  mass_flag = 0;
  com_flag = 0;
  vcm_flag = 0;

  dynamic_group_allow = 1;
  dynamic_group = group->dynamic[igroup];
  dynamic_mols = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "mass") == 0) {
      request_mass();
      iarg++;
    } else if (strcmp(arg[iarg], "com") == 0) {
      request_com();
      iarg++;
    } else if (strcmp(arg[iarg], "dynamic") == 0) {
      if (++iarg >= narg)
        error->all(FLERR, "Illegal fix property/mol command. "
            "Expected value after 'dynamic' keyword");
      dynamic_mols = utils::logical(FLERR, arg[iarg], false, lmp);
      iarg++;
    } else error->all(FLERR, "Illegal fix property/mol command");
  }

  nmax = 0;
  molmax = 1;
  nmolecule = 0;

  com_step = -1;
  mass_step = -1;
  vcm_step = -1;
  count_step = -1;

  array_flag = 1;
  size_array_cols = 4;
  size_array_rows_variable = 1;
}

/* ---------------------------------------------------------------------- */

FixPropertyMol::~FixPropertyMol()
{
  if (com_flag)
  {
    memory->destroy(com);
    memory->destroy(comproc);
  }
  if (vcm_flag)
  {
    memory->destroy(vcm);
    memory->destroy(vcmproc);
  }
  if (mass_flag)
  {
    memory->destroy(mass);
    memory->destroy(massproc);
  }
}

/* ---------------------------------------------------------------------- */

int FixPropertyMol::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  return mask;
}

void FixPropertyMol::request_com() {
  if (com_flag) return;
  com_flag = 1;
  request_mass();
  memory->create(com, nmax, 3, "property/mol:com");
  memory->create(comproc, nmax, 3, "property/mol:comproc");
}

void FixPropertyMol::request_vcm() {
  if (vcm_flag) return;
  vcm_flag = 1;
  ke_singles_flag = 1; // TODO EVK: need to find a better way to set this
  request_mass();
  memory->create(vcm, nmax, 3, "property/mol:vcm");
  memory->create(vcmproc, nmax, 3, "property/mol:vcmproc");
}

void FixPropertyMol::request_mass() {
  if (mass_flag) return;
  mass_flag = 1;
  memory->create(mass, nmax, "property/mol:mass");
  memory->create(massproc, nmax, "property/mol:massproc");
}

/* ---------------------------------------------------------------------- */

void FixPropertyMol::init()
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
void FixPropertyMol::setup_pre_force(int /*vflag*/) {
  dynamic_group = group->dynamic[igroup];
  grow_permolecule();

  // com_compute also computes mass if dynamic_group is set
  // so no need to call mass_compute in that case
  if (mass_flag && !dynamic_group) mass_compute();
  if (com_flag) com_compute();
  if (vcm_flag) vcm_compute();
  if (mass_flag) count_molecules();
}

void FixPropertyMol::setup_pre_force_respa(int vflag, int ilevel) {
  if (ilevel == 0) setup_pre_force(vflag);
}


/* ----------------------------------------------------------------------
   Calculate number of molecules and grow permolecule arrays if needed.
   Grows to the maximum of previous max. mol id + grow_by and new max. mol id
   if either is larger than nmax.
   Returns true if the number of molecules (max. mol id) changed.
------------------------------------------------------------------------- */

bool FixPropertyMol::grow_permolecule(int grow_by) {
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
      memory->grow(mass, nmax, "property/mol:mass");
      memory->grow(massproc, nmax, "property/mol:massproc");
    }
    if (com_flag)
    {
      memory->grow(com, nmax, 3, "property/mol:com");
      memory->grow(comproc, nmax, 3, "property/mol:comproc");
    }
    if(vcm_flag)
    {
      memory->grow(vcm, nmax, 3, "property/mol:vcm");
      memory->grow(vcmproc, nmax, 3, "property/mol:vcmproc");
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
void FixPropertyMol::count_molecules() {
  count_step = update->ntimestep;
  nmolecule = 0;
  for (tagint m = 0; m < molmax; ++m)
    if (mass[m] > 0.0) ++nmolecule;
}

/* ----------------------------------------------------------------------
   Update total mass of each molecule
------------------------------------------------------------------------- */

void FixPropertyMol::mass_compute() {
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

void FixPropertyMol::com_compute() {
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
   Calculate center of mass velocity of each molecule 
   Also update molecular mass if group is dynamic
------------------------------------------------------------------------- */
void FixPropertyMol::vcm_compute()
{
  vcm_step = update->ntimestep;
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
  double **v = atom->v;
  double massone;

  for (int m = 0; m < molmax; ++m) {
    vcmproc[m][0] = 0.0;
    vcmproc[m][1] = 0.0;
    vcmproc[m][2] = 0.0;
  }

  if (recalc_mass) {
    mass_step = update->ntimestep;
    for (tagint m = 0; m < molmax; ++m)
      massproc[m] = 0.0;
  }
  double ke_local[6];
  if (ke_singles_flag)
  {
    for (int i = 0; i < 6; ++i)
      ke_local[i] = 0.0;
  }

  for (int i = 0; i < nlocal; ++i) {
    if (groupbit & atom->mask[i]) {
      if (rmass) massone = rmass[i];
      else massone = amass[type[i]];
      tagint m = molecule[i]-1;
      if (m < 0) {
        if (ke_singles_flag) {
          ke_local[0] += v[i][0]*v[i][0]*massone;
          ke_local[1] += v[i][1]*v[i][1]*massone;
          ke_local[2] += v[i][2]*v[i][2]*massone;
          ke_local[3] += v[i][0]*v[i][1]*massone;
          ke_local[4] += v[i][0]*v[i][2]*massone;
          ke_local[5] += v[i][1]*v[i][2]*massone;
        }
        continue;
      }
      vcmproc[m][0] += v[i][0] * massone;
      vcmproc[m][1] += v[i][1] * massone;
      vcmproc[m][2] += v[i][2] * massone;
      if (recalc_mass) massproc[m] += massone;
    }
  }

  MPI_Allreduce(&vcmproc[0][0],&vcm[0][0],3*molmax,MPI_DOUBLE,MPI_SUM,world);
  if(ke_singles_flag) MPI_Allreduce(ke_local,ke_singles,6,MPI_DOUBLE, MPI_SUM, world);
  if (recalc_mass) MPI_Allreduce(massproc,mass,molmax,MPI_DOUBLE,MPI_SUM,world);

  for (int m = 0; m < molmax; ++m) {
    // Some molecule ids could be skipped (not assigned atoms)
    if (mass[m] > 0.0) {
      vcm[m][0] /= mass[m];
      vcm[m][1] /= mass[m];
      vcm[m][2] /= mass[m];
    } else {
      vcm[m][0] = vcm[m][1] = vcm[m][2] = 0.0;
    }
  }

}


/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixPropertyMol::memory_usage()
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

double FixPropertyMol::compute_array(int imol, int col)
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
    // TODO EVK: What about VCM?
  }
}
