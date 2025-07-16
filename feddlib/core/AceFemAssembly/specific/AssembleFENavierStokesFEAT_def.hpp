#ifndef ASSEMBLEFENAVIERSTOKESFEAT_DEF_hpp
#define ASSEMBLEFENAVIERSTOKESFEAT_DEF_hpp

#include "AssembleFENavierStokesFEAT_decl.hpp"
#include "feddlib/core/AceFemAssembly/specific/AssembleFENavierStokes_decl.hpp"

namespace FEDD {

template <class SC, class LO, class GO, class NO>
AssembleFENavierStokesFEAT<SC, LO, GO, NO>::AssembleFENavierStokesFEAT(int flag, vec2D_dbl_Type nodesRefConfig,
                                                                       ParameterListPtr_Type params,
                                                                       tuple_disk_vec_ptr_Type tuple)
    : AssembleFENavierStokes<SC, LO, GO, NO>(flag, nodesRefConfig, params, tuple) {
    auto fn = params->sublist("Parameter").get<std::function<void(SC *, const SC *, const SC *)>>("feat3 call-back");

    const int num_loc_entries = 12;
    const int num_loc_verts = 3;

    SC *mat = new SC[num_loc_entries * num_loc_entries];
    SC *loc_conv = new SC[num_loc_entries];
    std::fill(&loc_conv[0], &(loc_conv[0]) + num_loc_entries, SC(1.));
    SC *vt = new SC[num_loc_verts * 3]; // Allocate space for 3D coordinates
    SC verts[3][3] = {{0, 0, 0}, {0.1, 0, 0}, {0, 0.1, 0}};
    for (int i = 0; i < num_loc_verts; ++i) {
        SC *loc_vert = verts[i];
        for (int k = 0; k < 3; ++k) {
            vt[i * 3 + k] = loc_vert[k];
        }
    }

    fn(mat, loc_conv, vt);

    string output = "Loc Mat element 0: \n";

    for (int i = 0; i < num_loc_entries; ++i) {
        for (int j = 0; j < num_loc_entries; ++j) {
            output += std::to_string(mat[i * num_loc_entries + j]) + ", ";
        }
        output += "\n";
    }
    cout << output << endl;

    delete[] vt;
    delete[] loc_conv;
    delete[] mat;
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

    int dim = this->getDim();
    int numNodes = this->numNodesVelocity_;
    UN Grad = 2; // Needs to be fixed
    string FEType = this->FETypeVelocity_;
    int dofs = this->dofsVelocity_;

    vec3D_dbl_ptr_Type dPhi;
    vec_dbl_ptr_Type weights = Teuchos::rcp(new vec_dbl_Type(0));

    UN deg = Helper::determineDegree(dim, FEType, Grad);
    // cout << " Degree " << deg << " Grad " << Grad << " FeType " << FEType << endl;
    Helper::getDPhi(dPhi, weights, dim, FEType, deg);

    SC detB;
    SC absDetB;
    SmallMatrix<SC> B(dim);
    SmallMatrix<SC> Binv(dim);

    this->buildTransformation(B);

    detB = B.computeInverse(Binv);
    absDetB = std::fabs(detB);

    vec3D_dbl_Type dPhiTrans(dPhi->size(), vec2D_dbl_Type(dPhi->at(0).size(), vec_dbl_Type(dim, 0.)));
    Helper::applyBTinv(dPhi, dPhiTrans, Binv);

    for (UN i = 0; i < numNodes; i++) {
        Teuchos::Array<SC> value(dPhiTrans[0].size(), 0.);
        for (UN j = 0; j < numNodes; j++) {
            for (UN w = 0; w < dPhiTrans.size(); w++) {
                for (UN d = 0; d < dim; d++) {
                    value[j] += weights->at(w) * dPhiTrans[w][i][d] * dPhiTrans[w][j][d];
                }
            }
            value[j] *= absDetB;
            /*if (std::fabs(value[j]) < pow(10,-14)) {
               value[j] = 0.;
           }*/
            for (UN d = 0; d < dofs; d++) {
                (*elementMatrix)[i * dofs + d][j * dofs + d] = value[j];
            }
        }
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
