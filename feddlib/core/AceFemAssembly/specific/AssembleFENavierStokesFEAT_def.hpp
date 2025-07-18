#ifndef ASSEMBLEFENAVIERSTOKESFEAT_DEF_hpp
#define ASSEMBLEFENAVIERSTOKESFEAT_DEF_hpp

#include "AssembleFENavierStokesFEAT_decl.hpp"
#include "feddlib/core/AceFemAssembly/specific/AssembleFENavierStokes_decl.hpp"
#include <Teuchos_Assert.hpp>
#include <Teuchos_TestForException.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace FEDD {

template <class SC, class LO, class GO, class NO>
AssembleFENavierStokesFEAT<SC, LO, GO, NO>::AssembleFENavierStokesFEAT(int flag, vec2D_dbl_Type nodesRefConfig,
                                                                       ParameterListPtr_Type params,
                                                                       tuple_disk_vec_ptr_Type tuple)
    : AssembleFENavierStokes<SC, LO, GO, NO>(flag, nodesRefConfig, params, tuple) {
    featCallback_ =
        params->sublist("Parameter").get<std::function<void(SC *, const SC *, const SC *)>>("feat3 call-back");
    const int numLocEntries = this->numNodesVelocity_ * this->dofsVelocity_;
    featMat_ = std::vector<SC>(numLocEntries * numLocEntries);
    // TODO:[KH] what does this do? Can we hardcode it feat?
    locConv_ = std::vector<SC>(numLocEntries, 1);

    int numVerts;
    // TODO: [KH] for now only consider 2D
    TEUCHOS_TEST_FOR_EXCEPTION(this->getDim() != 2, std::runtime_error, "feat interface currently only supports 2D");

    if (this->FETypeVelocity_ == "P2") {
        if (this->getDim() == 2) {
            numVerts = 3;
        } else {
            numVerts = 4;
        }
    } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Unsupported FE type used for feat3 interface");
    }

    // feat expects 3D coords even in 2D, in which case the 3rd coord is zero.
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

    SmallMatrixPtr_Type elementMatrixN =
        Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
    SmallMatrixPtr_Type elementMatrixW =
        Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

    if (this->newtonStep_ == 0) {

        SmallMatrixPtr_Type elementMatrixA =
            Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
        SmallMatrixPtr_Type elementMatrixB =
            Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

        this->constantMatrix_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

        assemblyLaplacian(elementMatrixA);

        elementMatrixA->scale(this->viscosity_);
        elementMatrixA->scale(this->density_);

        this->constantMatrix_->add((*elementMatrixA), (*this->constantMatrix_));

        this->assemblyDivAndDivT(elementMatrixB); // For Matrix B

        elementMatrixB->scale(-1.);

        this->constantMatrix_->add((*elementMatrixB), (*this->constantMatrix_));
    }

    this->ANB_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_)); // A + B + N
    this->ANB_->add((*this->constantMatrix_), (*this->ANB_));

    assemblyAdvection(elementMatrixN);
    elementMatrixN->scale(this->density_);
    this->ANB_->add((*elementMatrixN), (*this->ANB_));
    if (this->linearization_ != "FixedPoint") {
        assemblyAdvectionInU(elementMatrixW);
        elementMatrixW->scale(this->density_);
    }

    // elementMatrix->add((*constantMatrix_),(*elementMatrix));
    this->jacobian_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

    this->jacobian_->add((*this->ANB_), (*this->jacobian_));
    // If the linearization is Newtons Method we need to add W-Matrix
    if (this->linearization_ != "FixedPoint") {
        this->jacobian_->add((*elementMatrixW),
                             (*this->jacobian_)); // int add(SmallMatrix<T> &bMat, SmallMatrix<T> &cMat); //this+B=C
                                                  // elementMatrix + constantMatrix_;
    }
}

template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::assembleFixedPoint() {

    SmallMatrixPtr_Type elementMatrixN =
        Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

    if (this->newtonStep_ == 0) {
        SmallMatrixPtr_Type elementMatrixA =
            Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));
        SmallMatrixPtr_Type elementMatrixB =
            Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

        this->constantMatrix_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

        assemblyLaplacian(elementMatrixA);

        elementMatrixA->scale(this->viscosity_);
        elementMatrixA->scale(this->density_);

        this->constantMatrix_->add((*elementMatrixA), (*this->constantMatrix_));

        this->assemblyDivAndDivT(elementMatrixB); // For Matrix B

        elementMatrixB->scale(-1.);

        this->constantMatrix_->add((*elementMatrixB), (*this->constantMatrix_));
    }

    this->ANB_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_)); // A + B + N
    this->ANB_->add((*this->constantMatrix_), (*this->ANB_));

    assemblyAdvection(elementMatrixN);
    elementMatrixN->scale(this->density_);
    this->ANB_->add((*elementMatrixN), (*this->ANB_));
}

template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::assemblyLaplacian(SmallMatrixPtr_Type &elementMatrix) {

    // num rows/cols of the element matrix
    LO numLocEntries = this->numNodesVelocity_ * this->dofsVelocity_;

    // Ask feat for the assembled element matrix
    featCallback_(featMat_.data(), locConv_.data(), verts4feat_.data());

    // Copy to fedd element matrix
    for (int i = 0; i < numLocEntries; i++) {
        std::copy(featMat_.begin() + i * numLocEntries, featMat_.begin() + (i + 1) * numLocEntries,
                  elementMatrix->getRow(i).begin());
    }
}

// Assemble RHS with updated solution coming from Fixed Point Iter or der Newton.
template <class SC, class LO, class GO, class NO> void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::assembleRHS() {

    SmallMatrixPtr_Type elementMatrixN =
        Teuchos::rcp(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_));

    this->ANB_.reset(new SmallMatrix_Type(this->dofsElementVelocity_ + this->numNodesPressure_)); // A + B + N
    this->ANB_->add((*this->constantMatrix_), (*this->ANB_));

    assemblyAdvection(elementMatrixN);
    elementMatrixN->scale(this->density_);
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
void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::assemblyAdvection(SmallMatrixPtr_Type &elementMatrix) {

    int dim = this->getDim();
    int numNodes = this->numNodesVelocity_;
    UN Grad = 2; // Needs to be fixed
    string FEType = this->FETypeVelocity_;
    int dofs = this->dofsVelocity_;

    vec3D_dbl_ptr_Type dPhi;
    vec2D_dbl_ptr_Type phi;
    vec_dbl_ptr_Type weights = Teuchos::rcp(new vec_dbl_Type(0));

    UN deg = Helper::determineDegree(dim, FEType, Grad); // Not complete
    // UN extraDeg = determineDegree( dim, FEType, Std); //Elementwise assembly of grad u
    // UN deg = determineDegree( dim, FEType, FEType, Grad, Std, extraDeg);

    Helper::getDPhi(dPhi, weights, dim, FEType, deg);
    Helper::getPhi(phi, weights, dim, FEType, deg);

    SC detB;
    SC absDetB;
    SmallMatrix<SC> B(dim);
    SmallMatrix<SC> Binv(dim);

    vec2D_dbl_Type uLoc(dim, vec_dbl_Type(weights->size(), -1.));

    this->buildTransformation(B);
    detB = B.computeInverse(Binv);
    absDetB = std::fabs(detB);

    vec3D_dbl_Type dPhiTrans(dPhi->size(), vec2D_dbl_Type(dPhi->at(0).size(), vec_dbl_Type(dim, 0.)));
    Helper::applyBTinv(dPhi, dPhiTrans, Binv);

    for (int w = 0; w < phi->size(); w++) { // quads points
        for (int d = 0; d < dim; d++) {
            uLoc[d][w] = 0.;
            for (int i = 0; i < phi->at(0).size(); i++) {
                LO index = dim * i + d;
                uLoc[d][w] += (*this->solution_)[index] * phi->at(w).at(i);
            }
        }
    }

    for (UN i = 0; i < phi->at(0).size(); i++) {
        Teuchos::Array<SC> value(dPhiTrans[0].size(), 0.);
        Teuchos::Array<GO> indices(dPhiTrans[0].size(), 0);
        for (UN j = 0; j < value.size(); j++) {
            for (UN w = 0; w < dPhiTrans.size(); w++) {
                for (UN d = 0; d < dim; d++) {
                    value[j] += weights->at(w) * uLoc[d][w] * (*phi)[w][i] * dPhiTrans[w][j][d];
                }
            }
            value[j] *= absDetB;

            /*if (setZeros_ && std::fabs(value[j]) < myeps_) {
                value[j] = 0.;
            }*/
        }
        for (UN d = 0; d < dim; d++) {
            for (UN j = 0; j < indices.size(); j++)
                (*elementMatrix)[i * dofs + d][j * dofs + d] = value[j];
        }
    }
}

template <class SC, class LO, class GO, class NO>
void AssembleFENavierStokesFEAT<SC, LO, GO, NO>::assemblyAdvectionInU(SmallMatrixPtr_Type &elementMatrix) {

    int dim = this->getDim();
    int numNodes = this->numNodesVelocity_;
    UN Grad = 2; // Needs to be fixed
    string FEType = this->FETypeVelocity_;
    int dofs = this->dofsVelocity_;

    vec3D_dbl_ptr_Type dPhi;
    vec2D_dbl_ptr_Type phi;
    vec_dbl_ptr_Type weights = Teuchos::rcp(new vec_dbl_Type(0));

    UN deg = Helper::determineDegree(dim, FEType, Grad); // Not complete
    // UN extraDeg = determineDegree( dim, FEType, Std); //Elementwise assembly of grad u
    // UN deg = determineDegree( dim, FEType, FEType, Grad, Std, extraDeg);

    Helper::getDPhi(dPhi, weights, dim, FEType, deg);
    Helper::getPhi(phi, weights, dim, FEType, deg);

    SC detB;
    SC absDetB;
    SmallMatrix<SC> B(dim);
    SmallMatrix<SC> Binv(dim);

    vec2D_dbl_Type uLoc(dim, vec_dbl_Type(weights->size(), -1.));

    this->buildTransformation(B);
    detB = B.computeInverse(Binv);
    absDetB = std::fabs(detB);

    vec3D_dbl_Type dPhiTrans(dPhi->size(), vec2D_dbl_Type(dPhi->at(0).size(), vec_dbl_Type(dim, 0.)));
    Helper::applyBTinv(dPhi, dPhiTrans, Binv);
    // UN FEloc = checkFE(dim,FEType);

    std::vector<SmallMatrix<SC>> duLoc(
        weights->size(),
        SmallMatrix<SC>(dim)); // for all quad points p_i each matrix is [u_x * grad Phi(p_i), u_y * grad Phi(p_i), u_z
                               // * grad Phi(p_i) (if 3D) ], duLoc[w] = [[phixx;phixy],[phiyx;phiyy]] (2D)

    for (int w = 0; w < dPhiTrans.size(); w++) { // quads points
        for (int d1 = 0; d1 < dim; d1++) {
            for (int i = 0; i < dPhiTrans[0].size(); i++) {
                LO index = dim * i + d1;
                for (int d2 = 0; d2 < dim; d2++)
                    duLoc[w][d2][d1] += (*this->solution_)[index] * dPhiTrans[w][i][d2];
            }
        }
    }

    for (UN i = 0; i < phi->at(0).size(); i++) {
        for (UN d1 = 0; d1 < dim; d1++) {
            Teuchos::Array<SC> value(dim * phi->at(0).size(), 0.); // These are value (W_ix,W_iy,W_iz)
            for (UN j = 0; j < phi->at(0).size(); j++) {
                for (UN d2 = 0; d2 < dim; d2++) {
                    for (UN w = 0; w < phi->size(); w++) {
                        value[dim * j + d2] += weights->at(w) * duLoc[w][d2][d1] * (*phi)[w][i] * (*phi)[w][j];
                    }
                    value[dim * j + d2] *= absDetB;
                }
            }
            for (UN j = 0; j < phi->at(0).size(); j++) {
                for (UN d = 0; d < dofs; d++) {
                    (*elementMatrix)[i * dofs + d1][j * dofs + d] = value[j * dofs + d];
                }
            }
        }
    }
}

} // namespace FEDD
#endif
