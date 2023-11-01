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

#include "fix_nvt_sllod_mol_kokkos.h"

#include "Kokkos_ExecPolicy.hpp"
#include "Kokkos_Macros.hpp"
#include "Kokkos_Parallel.hpp"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "compute.h"
#include "compute_temp_mol_kokkos.h"
#include "domain_kokkos.h"
#include "error.h"
#include "fix_deform_kokkos.h"
#include "fix_nh_kokkos.h"
#include "fix_property_mol_kokkos.h"
#include "group.h"
#include "kokkos_few.h"
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

template<class DeviceType>
FixNVTSllodMolKokkos<DeviceType>::FixNVTSllodMolKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixNHKokkos<DeviceType>(lmp, narg, arg), molpropKK(nullptr), id_molpropKK(nullptr)
{
  atomKK = (AtomKokkos *) this->atom;
  this->kokkosable = 1;
  this->domainKK = (DomainKokkos *) this->domain;

  molpropflag = 1;

  if (!this->tstat_flag)
    this->error->all(FLERR,"Temperature control must be used with fix nvt/sllod/mol/kk");
  if (this->pstat_flag)
    this->error->all(FLERR,"Pressure control can not be used with fix nvt/sllod/mol/kk");

  for (int iarg = 0; iarg < narg; ++iarg) {
    if (strcmp(arg[iarg], "^molprop")==0) {
      if (iarg+1 >= narg)
        this->error->all(FLERR,"Expected name of property/mol/kk fix after 'molprop'");
      this->id_molpropKK = utils::strdup(arg[iarg+1]);
      molpropflag = 0;
      iarg += 2;
    }
  }

  // default values

  if (this->mtchain_default_flag) {
    this->mtchain = 1;

    // Fix allocation of chain thermostats so that size_vector is correct
    int ich;
    delete[] this->eta;
    delete[] this->eta_dot;
    delete[] this->eta_dotdot;
    delete[] this->eta_mass;
    this->eta = new double[this->mtchain];

    // add one extra dummy thermostat, set to zero

    this->eta_dot = new double[this->mtchain+1];
    this->eta_dot[this->mtchain] = 0.0;
    this->eta_dotdot = new double[this->mtchain];
    for (ich = 0; ich < this->mtchain; ich++) {
      this->eta[ich] = this->eta_dot[ich] = this->eta_dotdot[ich] = 0.0;
    }
    this->eta_mass = new double[this->mtchain];

    // Default mtchain in fix_nh is 3.
    this->size_vector -= 2*2*(3-this->mtchain);
  }

  // create a new fix property/mol if needed
  // id = fix-ID + _molprop
  if (molpropflag) {
    this->id_molpropKK = utils::strdup(std::string(this->id) + "_molprop");
  }

  // create a new compute temp style
  // id = fix-ID + _temp
  this->id_temp = utils::strdup(std::string(this->id) + "_temp");
  this->modify->add_compute(fmt::format("{} {} temp/mol {}",
                      this->id_temp, this->group->names[this->igroup], id_molpropKK));
  this->tcomputeflag = 1;
}

/* ----------------------------------------------------------------------
   Create a fix property/mol if required
---------------------------------------------------------------------- */
template<class DeviceType>
void FixNVTSllodMolKokkos<DeviceType>::post_constructor() {
  if (molpropflag)
    this->modify->add_fix(fmt::format(
          "{} {} property/mol", id_molpropKK, this->group->names[this->igroup]));
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixNVTSllodMolKokkos<DeviceType>::~FixNVTSllodMolKokkos() {
  if (this->copymode) return;
  if (molpropflag && this->modify->nfix) this->modify->delete_fix(id_molpropKK);
  delete [] id_molpropKK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVTSllodMolKokkos<DeviceType>::init() {
  FixNHKokkos<DeviceType>::init();

  // Check that temperature calculates a molecular temperature
  // TODO(SS): add moltemp flag to compute.h that we can check?
  //if (strcmp(temperature->style, "temp/mol") != 0)
  //  error->all(FLERR,"fix nvt/sllod/mol requires temperature computed by "
  //      "compute temp/mol");

  // check fix deform remap settings

  int i;
  for (i = 0; i < this->modify->nfix; i++)
    if (strncmp(this->modify->fix[i]->style,"^deform",6) == 0) {
      auto def = dynamic_cast<FixDeformKokkos*>(this->modify->fix[i]);
      if (def->remapflag != Domain::NO_REMAP)
        this->error->all(FLERR,"Using fix nvt/sllod/mol with inconsistent fix deform "
                   "remap option");
      bool elongation = false;
      for (int j = 0; j < 3; ++j) {
        if (def->set[j].style) {
          elongation = true;
          if (def->set[j].style != TRATE)
            this->error->all(FLERR,"fix nvt/sllod/mol requires the trate style for "
                "x/y/z deformation");
        }
      }
      for (int j = 3; j < 6; ++j) {
        if (def->set[j].style && def->set[j].style != ERATE) {
          if (elongation)
            this->error->all(FLERR,"fix nvt/sllod/mol requires the erate style for "
                "xy/xz/yz deformation under mixed shear/extensional flow");
          else if (this->comm->me == 0)
            this->error->warning(FLERR,
                "Using non-constant shear rate with fix nvt/sllod/mol");
        }
      }
      if (this->comm->me == 0) {
        if (def->set[5].style && def->set[5].rate != 0.0 &&
            (def->set[3].style || domainKK->yz != 0.0) &&
            (def->set[4].style != ERATE || def->set[5].style != ERATE
             || (def->set[3].style && def->set[3].style != ERATE))
            )
          this->error->warning(FLERR,"Shearing xy with a yz tilt is only handled "
              "correctly if fix deform uses the erate style for xy, xz and yz");
        if (def->end_flag)
          this->error->warning(FLERR,"SLLOD equations of motion require box deformation"
              " to occur with position updates to be strictly correct. Set the N"
              " parameter of fix deform to 0 to enable this.");
      }
      break;
    }
  //TODO EVK: make sure the fix finds deform Kokkos as well
  //if (i == this->modify->nfix)
  //  this->error->all(FLERR,"Using fix nvt/sllod/mol with no fix deform defined");

  // Get id of molprop
  molpropKK = dynamic_cast<FixPropertyMolKokkos<DeviceType>*>(this->modify->get_fix_by_id(id_molpropKK));
  if (molpropKK == nullptr)
    this->error->all(FLERR, "Fix nvt/sllod/mol could not find a fix property/mol with id {}", id_molpropKK);
  // Make sure CoM and VCM can be computed
  molpropKK->request_com();
  molpropKK->request_vcm();

  // Check for exact group match since it's relied on for counting DoF by the temp compute
  if (this->igroup != molpropKK->igroup)
    this->error->all(FLERR, "Fix property/mol must be defined for the same group as fix nvt/sllod/mol");
}

/* ----------------------------------------------------------------------
   perform half-step scaling of velocities
-----------------------------------------------------------------------*/

template<class DeviceType>
void FixNVTSllodMolKokkos<DeviceType>::nh_v_temp() {
  // velocities stored as peculiar velocity (i.e. they don't include the SLLOD
  //   streaming velocity), so remove/restore bias will only be needed if some
  //   extra bias is being calculated.
  // thermostat thermal velocity only
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for temperature compute with BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias

  if (this->which == BIAS) {
    this->temperature->compute_scalar();
    this->temperature->remove_bias_all();
  }

  // Use molecular centre-of-mass velocity when calculating thermostat force
  molpropKK->vcm_compute();
  vcm = molpropKK->k_vcm.template view<DeviceType>();
  molID = atomKK->k_molecule.view<DeviceType>();
  int m;

  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (this->igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  double grad_u[6], vfac[3]; 
  MathExtra::multiply_shape_shape(domainKK->h_rate, domainKK->h_inv, grad_u);

  dt4 = 0.5*this->dthalf;
  vfac[0] = exp(-grad_u[0]*dt4);
  vfac[1] = exp(-grad_u[1]*dt4);
  vfac[2] = exp(-grad_u[2]*dt4);

  // Device copies
  d_grad_u = Few<double, 6>(grad_u);
  d_vfac = Few<double, 3>(vfac);

  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVTSllodMol_vtemp> (0, nlocal), *this);
  this->copymode = 0;
  /*
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & this->groupbit) {
      m = molID[i]-1;
      if (m < 0) vcm_local = v[i];  // CoM velocity of single atom is just v[i]
      else vcm_local = vcm[m];

      // First half step SLLOD force on CoM.
      // Don't overwrite vcm_local since we may need it for multiple atoms.
      vcm_local_new[0] = vcm_local[0]*vfac[0];
      vcm_local_new[1] = vcm_local[1]*vfac[1];
      vcm_local_new[2] = vcm_local[2]*vfac[2];
      vcm_local_new[1] -= dt4*grad_u[3]*vcm_local_new[2];
      vcm_local_new[0] -= dt4*(grad_u[5]*vcm_local_new[1] + grad_u[4]*vcm_local_new[2]);

      // Thermostat force
      vcm_local_new[0] *= this->factor_eta;
      vcm_local_new[1] *= this->factor_eta;
      vcm_local_new[2] *= this->factor_eta;

      // 2nd half step SLLOD force on CoM
      vcm_local_new[0] -= dt4*(grad_u[5]*vcm_local[1] + grad_u[4]*vcm_local[2]);
      vcm_local_new[1] -= dt4*grad_u[3]*vcm_local[2];
      vcm_local_new[0] *= vfac[0];
      vcm_local_new[1] *= vfac[1];
      vcm_local_new[2] *= vfac[2];

      // Update atom velocity with new CoM velocity
      v[i][0] = v[i][0] - vcm_local[0] + vcm_local_new[0];
      v[i][1] = v[i][1] - vcm_local[1] + vcm_local_new[1];
      v[i][2] = v[i][2] - vcm_local[2] + vcm_local_new[2];
    }
  }
  */

  if (this->which == BIAS) this->temperature->restore_bias_all();
  // v modified, need to synchronise
  atomKK->modified(this->execution_space, V_MASK);
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVTSllodMolKokkos<DeviceType>::operator()(TagFixNVTSllodMol_vtemp, const int &i) const {


  // Temporary variable for just this kernel
  // TODO EVK: this probably shouldn't be allocated inside the kernel. Should we just make one big
  // scratch array with dim (nlocal,3) up top?
  Few<double, 3> vcm_local_new;
  if (mask[i] & this->groupbit) {
      // Use a subview to keep track of the VCM to make the maths easier to keep track of. This
      // should be a shallow-copy so we can't overwrite it in case we need it for other atoms
      //
      // TODO EVK: Is there a way to defer declaring the type of vcm_local until we assign it?
      //           Using Kokkos's Subview type alias *should* work
      //           (https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Subviews.html), but
      //           I can't get clang to accept it as a valid type when assigning the subview
      //
      // Also TODO EVK: Apparently constructing a subview inside an inner loop is expensive, so
      //                maybe we should just directly use the vcm/v views and eat the code
      //                duplication messiness?
      //
      //using view_type = typename AT::t_v_array;
      //using subview_type = Kokkos::Subview<view_type,
      //                                  std::remove_const_t<decltype(Kokkos::ALL)>,
      //                                  Kokkos::pair<unsigned, unsigned>>;
      //subview_type vcm_local;

      tagint m = molID[i]-1;
      // TODO EVK: should probably put more thought into this so we can be more sure that all
      // threads within a warp will take the same branch. Is the list of atoms sorted so that
      // singles are at the end of the list? If not, can we do it and is it worth the extra cost?
      if (m < 0) { // Single atom
        auto vcm_local = Kokkos::subview(v, i, Kokkos::ALL);
        // First half step SLLOD force on CoM.
        // Don't overwrite vcm_local since we may need it for multiple atoms.
        vcm_local_new[0] = vcm_local[0]*d_vfac[0];
        vcm_local_new[1] = vcm_local[1]*d_vfac[1];
        vcm_local_new[2] = vcm_local[2]*d_vfac[2];
        vcm_local_new[1] -= dt4*d_grad_u[3]*vcm_local_new[2];
        vcm_local_new[0] -= dt4*(d_grad_u[5]*vcm_local_new[1] + d_grad_u[4]*vcm_local_new[2]);

        // Thermostat force
        vcm_local_new[0] *= this->factor_eta;
        vcm_local_new[1] *= this->factor_eta;
        vcm_local_new[2] *= this->factor_eta;

        // 2nd half step SLLOD force on CoM
        vcm_local_new[0] -= dt4*(d_grad_u[5]*vcm_local[1] + d_grad_u[4]*vcm_local[2]);
        vcm_local_new[1] -= dt4*d_grad_u[3]*vcm_local[2];
        vcm_local_new[0] *= d_vfac[0];
        vcm_local_new[1] *= d_vfac[1];
        vcm_local_new[2] *= d_vfac[2];

        // Update atom velocity with new CoM velocity
        v(i, 0) = v(i, 0) - vcm_local[0] + vcm_local_new[0];
        v(i, 1) = v(i, 1) - vcm_local[1] + vcm_local_new[1];
        v(i, 2) = v(i, 2) - vcm_local[2] + vcm_local_new[2];

      } else { // Part of a molecule
        auto vcm_local = Kokkos::subview(vcm, m, Kokkos::ALL);

        // First half step SLLOD force on CoM.
        // Don't overwrite vcm_local since we may need it for multiple atoms.
        vcm_local_new[0] = vcm_local[0]*d_vfac[0];
        vcm_local_new[1] = vcm_local[1]*d_vfac[1];
        vcm_local_new[2] = vcm_local[2]*d_vfac[2];
        vcm_local_new[1] -= dt4*d_grad_u[3]*vcm_local_new[2];
        vcm_local_new[0] -= dt4*(d_grad_u[5]*vcm_local_new[1] + d_grad_u[4]*vcm_local_new[2]);

        // Thermostat force
        vcm_local_new[0] *= this->factor_eta;
        vcm_local_new[1] *= this->factor_eta;
        vcm_local_new[2] *= this->factor_eta;

        // 2nd half step SLLOD force on CoM
        vcm_local_new[0] -= dt4*(d_grad_u[5]*vcm_local[1] + d_grad_u[4]*vcm_local[2]);
        vcm_local_new[1] -= dt4*d_grad_u[3]*vcm_local[2];
        vcm_local_new[0] *= d_vfac[0];
        vcm_local_new[1] *= d_vfac[1];
        vcm_local_new[2] *= d_vfac[2];

        // Update atom velocity with new CoM velocity
        v(i, 0) = v(i, 0) - vcm_local[0] + vcm_local_new[0];
        v(i, 1) = v(i, 1) - vcm_local[1] + vcm_local_new[1];
        v(i, 2) = v(i, 2) - vcm_local[2] + vcm_local_new[2];
      }
    }
}


/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

template<class DeviceType>
void FixNVTSllodMolKokkos<DeviceType>::nve_x()
{
  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  molID = atomKK->k_molecule.view<DeviceType>();
  image = atomKK->k_image.view<DeviceType>();
  tagint m;

  double grad_u[6], xfac[3];
  dtv2 = this->dtv*0.5;
  int nlocal = atomKK->nlocal;

  double *xcom, xcom_half[3], molcom[3];
  com = molpropKK->k_com.template view<DeviceType>();

  if (this->igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  // x update by full step only for atoms in group

  double* h_rate = domainKK->h_rate;
  double* h_inv = domainKK->h_inv;
  MathExtra::multiply_shape_shape(h_rate, h_inv, grad_u);
  xfac[0] = exp(grad_u[0]*dtv2);
  xfac[1] = exp(grad_u[1]*dtv2);
  xfac[2] = exp(grad_u[2]*dtv2);
  // Device copies
  d_grad_u = Few<double, 6>(grad_u);
  d_xfac = Few<double, 3>(xfac);
  // Get the box properties from domainKK so we can remap inside the kernel
  prd = Few<double, 3>(domainKK->prd);
  h = Few<double, 6>(domainKK->h);
  triclinic = domainKK->triclinic;

  // Calculate CoM
  molpropKK->com_compute();

  // First half step
  /*
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      m = molID[i]-1;
      if (m < 0) xcom = x[i];
      else {
        // CoM stored in unwrapped coords.
        // Need to wrap to same image as x[i] streaming velocity is correct.
        molcom[0] = com[m][0];
        molcom[1] = com[m][1];
        molcom[2] = com[m][2];
        // Use inverted sign of atom's image to map CoM to correct position
        imageint ix = (2*IMGMAX - (atom->image[i] & IMGMASK)) & IMGMASK;
        imageint iy = (2*IMGMAX - (atom->image[i] >> IMGBITS & IMGMASK)) & IMGMASK;
        imageint iz = (2*IMGMAX - (atom->image[i] >> IMG2BITS)) & IMGMASK;
        domainKK->unmap(molcom, ix | (iy << IMGBITS) | (iz << IMG2BITS));
        xcom = molcom;
      }

      xcom_half[0] = xcom[0]*xfac[0];
      xcom_half[1] = xcom[1]*xfac[1];
      xcom_half[2] = xcom[2]*xfac[2];
      xcom_half[1] += dtv2*grad_u[3]*xcom_half[2];
      xcom_half[0] += dtv2*(grad_u[5]*xcom_half[1] + grad_u[4]*xcom_half[2]);

      x[i][0] = x[i][0] - xcom[0] + xcom_half[0] + dtv*v[i][0];
      x[i][1] = x[i][1] - xcom[1] + xcom_half[1] + dtv*v[i][1];
      x[i][2] = x[i][2] - xcom[2] + xcom_half[2] + dtv*v[i][2];
    }
  }
  */
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVTSllodMol_x1> (0, nlocal), *this);
  this->copymode = 0;

  // x modified, need to synchronise
  // TODO EVK: Don't actually think we do, since molprop Kokkos just uses the device views
  // everywhere
  atomKK->modified(this->execution_space, X_MASK);

  // Update CoM
  molpropKK->com_compute();

  /*
  // 2nd reversible half step
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      m = molID[i]-1;
      if (m < 0) xcom = x[i];
      else {
        molcom[0] = com[m][0];
        molcom[1] = com[m][1];
        molcom[2] = com[m][2];
        imageint ix = (2*IMGMAX - (atom->image[i] & IMGMASK)) & IMGMASK;
        imageint iy = (2*IMGMAX - (atom->image[i] >> IMGBITS & IMGMASK)) & IMGMASK;
        imageint iz = (2*IMGMAX - (atom->image[i] >> IMG2BITS)) & IMGMASK;
        domain->unmap(molcom, ix | (iy << IMGBITS) | (iz << IMG2BITS));
        xcom = molcom;
      }

      xcom_half[0] = xcom[0] + dtv2*(grad_u[5]*xcom[1] + grad_u[4]*xcom[2]);
      xcom_half[1] = xcom[1] + dtv2*grad_u[3]*xcom[2];
      xcom_half[2] = xcom[2]*xfac[2];
      xcom_half[1] *= xfac[1];
      xcom_half[0] *= xfac[0];

      x[i][0] = x[i][0] - xcom[0] + xcom_half[0];
      x[i][1] = x[i][1] - xcom[1] + xcom_half[1];
      x[i][2] = x[i][2] - xcom[2] + xcom_half[2];
    }
  }
  */
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVTSllodMol_x2> (0, nlocal), *this);
  this->copymode = 0;
  // x modified, need to synchronise again
  atomKK->modified(this->execution_space, X_MASK);
}

template <class DeviceType>
void FixNVTSllodMolKokkos<DeviceType>::operator()(TagFixNVTSllodMol_x1, const int &i) const {
  // Temporary variable for just this kernel
  // TODO EVK: this probably shouldn't be allocated inside the kernel. Should we just make one big
  // scratch array with dim (nlocal,3) up top?
  Few<double, 3> xcom_half;
  if (mask[i] & this->groupbit) {
      int m = molID[i]-1;
      // TODO EVK: should probably put more thought into this so we can be more sure that all
      // threads within a warp will take the same branch. Is the list of atoms sorted so that
      // singles are at the end of the list? If not, can we do it and is it worth the extra cost?
      if (m < 0) {
        auto xcom = Kokkos::subview(x, i, Kokkos::ALL);
        xcom_half[0] = xcom[0]*d_xfac[0];
        xcom_half[1] = xcom[1]*d_xfac[1];
        xcom_half[2] = xcom[2]*d_xfac[2];
        xcom_half[1] += dtv2*d_grad_u[3]*xcom_half[2];
        xcom_half[0] += dtv2*(d_grad_u[5]*xcom_half[1] + d_grad_u[4]*xcom_half[2]);

        x(i, 0) = x(i, 0) - xcom[0] + xcom_half[0] + this->dtv2*v(i, 0);
        x(i, 1) = x(i, 1) - xcom[1] + xcom_half[1] + this->dtv2*v(i, 1);
        x(i, 2) = x(i, 2) - xcom[2] + xcom_half[2] + this->dtv2*v(i, 2);
      } else {
        //double unwrapped[3];
        //double com_i[3];
        Few<double, 3> unwrapped, com_i;
        // CoM stored in unwrapped coords.
        // Need to wrap to same image as x[i] streaming velocity is correct.
        com_i[0] = com(m, 0);
        com_i[1] = com(m, 1);
        com_i[2] = com(m, 2);
        // Use inverted sign of atom's image to map CoM to correct position
        imageint ix = (2*IMGMAX - (image[i] & IMGMASK)) & IMGMASK;
        imageint iy = (2*IMGMAX - (image[i] >> IMGBITS & IMGMASK)) & IMGMASK;
        imageint iz = (2*IMGMAX - (image[i] >> IMG2BITS)) & IMGMASK;
        unwrapped = DomainKokkos::unmap(prd, h, triclinic, com_i, image[i]);

        xcom_half[0] = unwrapped[0]*d_xfac[0];
        xcom_half[1] = unwrapped[1]*d_xfac[1];
        xcom_half[2] = unwrapped[2]*d_xfac[2];
        xcom_half[1] += dtv2*d_grad_u[3]*xcom_half[2];
        xcom_half[0] += dtv2*(d_grad_u[5]*xcom_half[1] + d_grad_u[4]*xcom_half[2]);

        x(i, 0) = x(i, 0) - unwrapped[0] + xcom_half[0] + this->dtv*v(i, 0);
        x(i, 1) = x(i, 1) - unwrapped[1] + xcom_half[1] + this->dtv*v(i, 1);
        x(i, 2) = x(i, 2) - unwrapped[2] + xcom_half[2] + this->dtv*v(i, 2);

    }
  }
}

template <class DeviceType>
void FixNVTSllodMolKokkos<DeviceType>::operator()(TagFixNVTSllodMol_x2, const int &i) const {

  Few<double, 3> xcom_half;
  if (mask[i] & this->groupbit) {
    int m = molID[i]-1;
      if (m < 0) {
        auto xcom = Kokkos::subview(x, i, Kokkos::ALL);
        xcom_half[0] = xcom[0] + dtv2*(d_grad_u[5]*xcom[1] + d_grad_u[4]*xcom[2]);
        xcom_half[1] = xcom[1] + dtv2*d_grad_u[3]*xcom[2];
        xcom_half[2] = xcom[2]*d_xfac[2];
        xcom_half[1] *= d_xfac[1];
        xcom_half[0] *= d_xfac[0];

        x(i, 0) = x(i, 0) - xcom[0] + xcom_half[0];
        x(i, 1) = x(i, 1) - xcom[1] + xcom_half[1];
        x(i, 2) = x(i, 2) - xcom[2] + xcom_half[2];
      } else {
        Few<double, 3> unwrapped, com_i;
        com_i[0] = com(m, 0);
        com_i[1] = com(m, 1);
        com_i[2] = com(m, 2);
        imageint ix = (2*IMGMAX - (image[i] & IMGMASK)) & IMGMASK;
        imageint iy = (2*IMGMAX - (image[i] >> IMGBITS & IMGMASK)) & IMGMASK;
        imageint iz = (2*IMGMAX - (image[i] >> IMG2BITS)) & IMGMASK;
        unwrapped = DomainKokkos::unmap(prd, h, triclinic, com_i, image[i]);

        xcom_half[0] = unwrapped[0] + dtv2*(d_grad_u[5]*unwrapped[1] + d_grad_u[4]*unwrapped[2]);
        xcom_half[1] = unwrapped[1] + dtv2*d_grad_u[3]*unwrapped[2];
        xcom_half[2] = unwrapped[2]*d_xfac[2];
        xcom_half[1] *= d_xfac[1];
        xcom_half[0] *= d_xfac[0];

        x(i, 0) = x(i, 0) - unwrapped[0] + xcom_half[0];
        x(i, 1) = x(i, 1) - unwrapped[1] + xcom_half[1];
        x(i, 2) = x(i, 2) - unwrapped[2] + xcom_half[2];
      }
    }
}

namespace LAMMPS_NS {
template class FixNVTSllodMolKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixNVTSllodMolKokkos<LMPHostType>;
#endif
}


