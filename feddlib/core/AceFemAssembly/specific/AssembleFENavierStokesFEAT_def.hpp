#ifndef ASSEMBLEFENAVIERSTOKESFEAT_DEF_hpp
#define ASSEMBLEFENAVIERSTOKESFEAT_DEF_hpp

#include "AssembleFENavierStokesFEAT_decl.hpp"
#include "feddlib/core/AceFemAssembly/specific/AssembleFENavierStokes_decl.hpp"
#include <Teuchos_Assert.hpp>
#include <Teuchos_TestForException.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace FEDD {

template <class SC, class LO, class GO, class NO>
AssembleFENavierStokesFEAT<SC, LO, GO, NO>::AssembleFENavierStokesFEAT(int flag, vec2D_dbl_Type nodesRefConfig,
                                                                       ParameterListPtr_Type params,
                                                                       tuple_disk_vec_ptr_Type tuple)
    : AssembleFENavierStokes<SC, LO, GO, NO>(flag, nodesRefConfig, params, tuple) {
    featDiffusion_ =
        params->sublist("Parameter").get<std::function<void(SC *, const SC *, const SC *)>>("feat3 diffusion");
    featAdvection_ =
        params->sublist("Parameter").get<std::function<void(SC *, const SC *, const SC *)>>("feat3 advection");
    featFrechetAdvection_ =
        params->sublist("Parameter").get<std::function<void(SC *, const SC *, const SC *)>>("feat3 frechet advection");

    featMat_ = std::vector<SC>(this->dofsElementVelocity_ * this->dofsElementVelocity_, 0.);
    locConv_ = std::vector<SC>(this->dofsElementVelocity_, 1.);

    int numVerts;

    if (this->FETypeVelocity_ == "P2") {
        if (this->getDim() == 2) {
            numVerts = 3;
        } else {
            numVerts = 4;
        }
    } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Unsupported FE type used for feat3 interface");
    }

    verts4feat_.resize(numVerts * 3);

    for (int i(0); i < numVerts; i++) {
        auto &vert = this->nodesRefConfig_[i];
        std::copy(vert.begin(), vert.end(), verts4feat_.begin() + i * 3);
        // feat uses 3D coords also in 2D, in which case the last coord is empty
        if (this->getDim() == 2) {
            verts4feat_[i * 3 + 2] = 0;
        }
    }
}

template <class SC, class LO, class GO, class NO> void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::assembleJacobian() {

    if (this->newtonStep_ == 0) {
        assembleConstantMatrix();
    }

    // Ask feat for the assembled element matrix
    std::copy(this->solution_->begin(), this->solution_->begin() + this->dofsElementVelocity_,
              this->solutionVelocity_.begin());
    featFrechetAdvection_(featMat_.data(), this->solutionVelocity_.data(), verts4feat_.data());

    SmallMatrixPtr_Type elementMatrixNW =
        Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

    // Copy to fedd element matrix
    copyFEAT2FEDD(elementMatrixNW);

    this->jacobian_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
    this->jacobian_->add((*this->constantMatrix_), (*this->jacobian_));
    this->jacobian_->add((*elementMatrixNW), (*this->jacobian_));
}

template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::assembleFixedPoint() {

    if (this->newtonStep_ == 0) {
        assembleConstantMatrix();
    }

    this->ANB_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_)); // A + B + N
    this->ANB_->add((*this->constantMatrix_), (*this->ANB_));

    SmallMatrixPtr_Type elementMatrixN =
        Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

    assemblyAdvection(elementMatrixN);
    elementMatrixN->scale(this->density_);
    this->ANB_->add((*elementMatrixN), (*this->ANB_));
}

// Assemble RHS with updated solution coming from Fixed Point Iter or der Newton.
template <class SC, class LO, class GO, class NO> void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::assembleRHS() {

    SmallMatrixPtr_Type elementMatrixN =
        Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

    assemblyAdvection(elementMatrixN);
    elementMatrixN->scale(this->density_);

    this->ANB_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_)); // A + B + N
    this->ANB_->add((*this->constantMatrix_), (*this->ANB_));
    this->ANB_->add((*elementMatrixN), (*this->ANB_));

    this->rhsVec_.reset(new vec_dbl_Type(this->dofsElement_, 0.));
    // Multiplying ANB_ * solution // ANB Matrix without nonlinear part.
    int s = 0, t = 0;
    for (int i = 0; i < this->ANB_->size(); i++) {
        if (i >= this->dofsElementVelocity_)
            s = 1;
        for (int j = 0; j < this->ANB_->size(); j++) {
            if (j >= this->dofsElementVelocity_)
                t = 1;
            (*this->rhsVec_)[i] += (*this->ANB_)[i][j] * (*this->solution_)[j] * this->coeff_[s][t];
        }
        t = 0;
    }
}

template <class SC, class LO, class GO, class NO>
inline void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::assemblyAdvection(SmallMatrixPtr_Type &elementMatrix) {

    // Ask feat for the assembled element matrix
    std::copy(this->solution_->begin(), this->solution_->begin() + this->dofsElementVelocity_,
              this->solutionVelocity_.begin());
    featAdvection_(featMat_.data(), this->solutionVelocity_.data(), verts4feat_.data());
    // Copy to fedd element matrix
    copyFEAT2FEDD(elementMatrix);
}

template <class SC, class LO, class GO, class NO>
inline void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::assembleConstantMatrix() {

    SmallMatrixPtr_Type elementMatrixA =
        Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
    SmallMatrixPtr_Type elementMatrixB =
        Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

    this->constantMatrix_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

    // Assemble vector Laplace with feat
    // Ask feat for the assembled element matrix
    featDiffusion_(featMat_.data(), locConv_.data(), verts4feat_.data());
    // Copy to fedd element matrix
    copyFEAT2FEDD(elementMatrixA);

    this->constantMatrix_->add((*elementMatrixA), (*this->constantMatrix_));

    // This is not done by feat (for now)
    this->assemblyDivAndDivT(elementMatrixB); // For Matrix B

    elementMatrixB->scale(-1.);

    this->constantMatrix_->add((*elementMatrixB), (*this->constantMatrix_));
}

template <class SC, class LO, class GO, class NO>
inline void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::copyFEAT2FEDD(SmallMatrixPtr_Type feddMat) const {
    for (int i = 0; i < this->dofsElementVelocity_; i++) {
        std::copy(featMat_.begin() + i * this->dofsElementVelocity_,
                  featMat_.begin() + (i + 1) * this->dofsElementVelocity_, feddMat->getRow(i).begin());
    }
}

} // namespace FEDD
#endif
