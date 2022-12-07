// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Emily Kahl (Uni of QLD)
------------------------------------------------------------------------- */

#include "fix_nvt_asllod_mol.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "compute_temp_mol.h"
#include "domain.h"
#include "error.h"
#include "fix_deform.h"
#include "fix_property_mol.h"
#include "group.h"
#include "math_extra.h"
#include "modify.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

// from FixNH:
enum{NOBIAS,BIAS};

// from FixDeform:
enum{NONE=0,FINAL,DELTA,SCALE,VEL,ERATE,TRATE,VOLUME,WIGGLE,VARIABLE};
/* ---------------------------------------------------------------------- */

FixNVTAsllodMol::FixNVTAsllodMol(LAMMPS *lmp, int narg, char **arg) :
  FixNH(lmp, narg, arg), id_molprop(nullptr), molprop(nullptr)
{
  molpropflag = 1;

  if (!tstat_flag)
    error->all(FLERR,"Temperature control must be used with fix nvt/a-sllod/mol");
  if (pstat_flag)
    error->all(FLERR,"Pressure control can not be used with fix nvt/a-sllod/mol");

  for (int iarg = 0; iarg < narg; ++iarg) {
    if (strcmp(arg[iarg], "molprop")==0) {
      if (iarg+1 >= narg)
        error->all(FLERR,"Expected name of property/mol fix after 'molprop'");
      id_molprop = utils::strdup(arg[iarg+1]);
      molpropflag = 0;
      iarg += 2;
    }
  }

  // default values

  if (mtchain_default_flag) {
    mtchain = 1;

    // Fix allocation of chain thermostats so that size_vector is correct
    int ich;
    delete[] eta;
    delete[] eta_dot;
    delete[] eta_dotdot;
    delete[] eta_mass;
    eta = new double[mtchain];

    // add one extra dummy thermostat, set to zero

    eta_dot = new double[mtchain+1];
    eta_dot[mtchain] = 0.0;
    eta_dotdot = new double[mtchain];
    for (ich = 0; ich < mtchain; ich++) {
      eta[ich] = eta_dot[ich] = eta_dotdot[ich] = 0.0;
    }
    eta_mass = new double[mtchain];

    // Default mtchain in fix_nh is 3.
    size_vector -= 2*2*(3-mtchain);
  }

  // create a new fix property/mol if needed
  // id = fix-ID + _molprop
  if (molpropflag) {
    id_molprop = utils::strdup(std::string(id) + "_molprop");
  }

  // create a new compute temp style
  // id = fix-ID + _temp
  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp/mol {}",
                      id_temp, group->names[igroup], id_molprop));
  tcomputeflag = 1;
}

/* ----------------------------------------------------------------------
   Create a fix property/mol if required
---------------------------------------------------------------------- */
void FixNVTAsllodMol::post_constructor() {
  if (molpropflag)
    modify->add_fix(fmt::format(
          "{} {} property/mol", id_molprop, group->names[igroup]));
}

/* ---------------------------------------------------------------------- */

FixNVTAsllodMol::~FixNVTAsllodMol() {
  if (molpropflag && modify->nfix) modify->delete_fix(id_molprop);
  delete [] id_molprop;
}

/* ---------------------------------------------------------------------- */

void FixNVTAsllodMol::init() {
  FixNH::init();

  // Get id of molprop
  molprop = dynamic_cast<FixPropertyMol*>(modify->get_fix_by_id(id_molprop));
  if (molprop == nullptr)
    error->all(FLERR, "Compute temp/mol could not find a fix property/mol with id {}", id_molprop);
  // if (!molprop->mass_flag)
  //   error->all(FLERR, "Compute temp/mol requires fix property/mol with the mass or com flag");
  if (igroup != molprop->igroup)
    error->all(FLERR, "Fix property/mol must be defined for the same group as compute temp/mol");
  // Register to get the VCM
  // TODO EVK: No need to pass ke_singles pointer since we only care about vcm
  // Need to find a way to avoid clashing with other styles that *do* need it
  molprop->request_vcm();

  // Check that temperature calculates a molecular temperature
  // TODO(SS): add moltemp flag to compute.h that we can check?
  if (strcmp(temperature->style, "temp/mol") != 0)
    error->all(FLERR,"fix nvt/a-sllod/mol requires temperature computed by "
        "compute temp/mol");

  // check fix deform remap settings

  int i;
  for (i = 0; i < modify->nfix; i++)
    if (strncmp(modify->fix[i]->style,"deform",6) == 0) {
      auto def = dynamic_cast<FixDeform *>(modify->fix[i]);
      if (def->remapflag != Domain::NO_REMAP)
        error->all(FLERR,"Using fix nvt/a-sllod/mol with inconsistent fix deform "
                   "remap option");
      bool elongation = false;
      for (int j = 0; j < 3; ++j) {
        if (def->set[j].style) {
          elongation = true;
          if (def->set[j].style != TRATE)
            error->all(FLERR,"fix nvt/a-sllod/mol requires the trate style for "
                "x/y/z deformation");
        }
      }
      for (int j = 3; j < 6; ++j) {
        if (def->set[j].style && def->set[j].style != ERATE) {
          if (elongation)
            error->all(FLERR,"fix nvt/a-sllod/mol requires the erate style for "
                "xy/xz/yz deformation under mixed shear/extensional flow");
          else if (comm->me == 0)
            error->warning(FLERR,
                "Using non-constant shear rate with fix nvt/sllod/mol");
        }
      }
      if (comm->me == 0) {
        if (def->set[5].style && def->set[5].rate != 0.0 &&
            (def->set[3].style || domain->yz != 0.0) &&
            (def->set[4].style != ERATE || def->set[5].style != ERATE
             || (def->set[3].style && def->set[3].style != ERATE))
            )
          error->warning(FLERR,"Shearing xy with a yz tilt is only handled "
              "correctly if fix deform uses the erate style for xy, xz and yz");
        if (def->end_flag)
          error->warning(FLERR,"SLLOD equations of motion require box deformation"
              " to occur with position updates to be strictly correct. Set the N"
              " parameter of fix deform to 0 to enable this.");
      }
      break;
    }
  if (i == modify->nfix)
    error->all(FLERR,"Using fix nvt/a-sllod/mol with no fix deform defined");

  // Check for exact group match since it's relied on for calculating CoM velocity
  if (igroup != temperature->igroup)
    error->all(FLERR,
        "Fix nvt/a-sllod/mol must be defined for the same group as compute temp/mol");
}

/* ----------------------------------------------------------------------
   perform half-step scaling of velocities
-----------------------------------------------------------------------*/

void FixNVTAsllodMol::nh_v_temp() {
  // velocities stored as peculiar velocity (i.e. they don't include the SLLOD
  //   streaming velocity), so remove/restore bias will only be needed if some
  //   extra bias is being calculated.
  // thermostat thermal velocity only
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for temperature compute with BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias

  if (which == BIAS) {
    temperature->compute_scalar();
    temperature->remove_bias_all();
  }

  // Use molecular centre-of-mass velocity when calculating thermostat force
  auto temp_mol = dynamic_cast<ComputeTempMol*>(temperature);
  tagint *molecule = atom->molecule;
  int m;

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double grad_u[6], vfac[3];
  double* h_rate = domain->h_rate;
  double* h_inv = domain->h_inv;
  MathExtra::multiply_shape_shape(h_rate, h_inv, grad_u);

  double dt4 = 0.5*dthalf;
  vfac[0] = exp(-grad_u[0]*dt4);
  vfac[1] = exp(-grad_u[1]*dt4);
  vfac[2] = exp(-grad_u[2]*dt4);

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // First half step SLLOD force on atoms
      v[i][0] *= vfac[0];
      v[i][1] *= vfac[1];
      v[i][2] *= vfac[2];
      v[i][1] -= dt4*grad_u[3]*v[i][2];
      v[i][0] -= dt4*(grad_u[5]*v[i][1] + grad_u[4]*v[i][2]);
    }
  }

  // Get mid-step CoM velocity
  double **vcm = molprop->vcm;
  double *vcom;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) vcom = v[i];  // CoM velocity of single atom is just v[i]
      else vcom = vcm[m];

      // Thermostat force on CoM velocity
      v[i][0] = v[i][0] - vcom[0] + vcom[0] * factor_eta;
      v[i][1] = v[i][1] - vcom[1] + vcom[1] * factor_eta;
      v[i][2] = v[i][2] - vcom[2] + vcom[2] * factor_eta;

      // 2nd half step SLLOD force on atoms
      v[i][0] -= dt4*(grad_u[5]*v[i][1] + grad_u[4]*v[i][2]);
      v[i][1] -= dt4*grad_u[3]*v[i][2];
      v[i][0] *= vfac[0];
      v[i][1] *= vfac[1];
      v[i][2] *= vfac[2];
    }
  }

  if (which == BIAS) temperature->restore_bias_all();
}


/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

void FixNVTAsllodMol::nve_x()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  double grad_u[6], xfac[3];
  double dtv2 = dtv*0.5;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // x update by full step only for atoms in group

  double* h_rate = domain->h_rate;
  double* h_inv = domain->h_inv;
  MathExtra::multiply_shape_shape(h_rate, h_inv, grad_u);
  xfac[0] = exp(grad_u[0]*dtv2);
  xfac[1] = exp(grad_u[1]*dtv2);
  xfac[2] = exp(grad_u[2]*dtv2);

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      x[i][0] *= xfac[0];
      x[i][1] *= xfac[1];
      x[i][2] *= xfac[2];
      x[i][1] += dtv2 * grad_u[3]*x[i][2];
      x[i][0] += dtv2 * (grad_u[5]*x[i][1] + grad_u[4]*x[i][2]);
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
      x[i][0] += dtv2 * (grad_u[5]*x[i][1] + grad_u[4]*x[i][2]);
      x[i][1] += dtv2 * grad_u[3]*x[i][2];
      x[i][0] *= xfac[0];
      x[i][1] *= xfac[1];
      x[i][2] *= xfac[2];
    }
  }
}
