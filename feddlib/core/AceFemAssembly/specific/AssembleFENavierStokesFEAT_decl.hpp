#ifndef ASSEMBLEFENAVIERSTOKESFEAT_DECL_hpp
#define ASSEMBLEFENAVIERSTOKESFEAT_DECL_hpp

#include "feddlib/core/AceFemAssembly/AssembleFE.hpp"
#include "feddlib/core/AceFemAssembly/specific/AssembleFENavierStokes.hpp"
#include "feddlib/core/AceFemAssembly/specific/AssembleFENavierStokes_decl.hpp"
#include "feddlib/core/FE/Helper.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/LinearAlgebra/Matrix.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include <vector>

namespace FEDD {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class AssembleFENavierStokesFEAT : public AssembleFENavierStokes<SC, LO, GO, NO> {
  public:
    typedef Matrix<SC, LO, GO, NO> Matrix_Type;
    typedef Teuchos::RCP<Matrix_Type> MatrixPtr_Type;

    typedef SmallMatrix<SC> SmallMatrix_Type;
    typedef Teuchos::RCP<SmallMatrix_Type> SmallMatrixPtr_Type;

    typedef MultiVector<SC, LO, GO, NO> MultiVector_Type;
    typedef Teuchos::RCP<MultiVector_Type> MultiVectorPtr_Type;

    typedef AssembleFE<SC, LO, GO, NO> AssembleFE_Type;

    /*!
     \brief Assemble the element Jacobian matrix.
    */
    void assembleJacobian() override;

    /*!
     \brief Assemble the element right hand side vector.
    */
    void assembleRHS() override;

    /*!
    \brief Assembly of FixedPoint- Matrix (System Matrix K with current u)
    */
    void assembleFixedPoint();

  protected:
    /*!

     \brief Constructor for AssembleFEAceNavierStokes

    @param[in] flag Flag of element
    @param[in] nodesRefConfig Nodes of element in reference configuration
    @param[in] params Parameterlist for current problem
    @param[in] tuple vector of element information tuples.
    */
    AssembleFENavierStokesFEAT(int flag, vec2D_dbl_Type nodesRefConfig, ParameterListPtr_Type parameters,
                               tuple_disk_vec_ptr_Type tuple);

    /*!

     \brief Assembly function for vector values laplacian \f$ \int_T \nabla v \cdot \nabla u ~dx\f$
    @param[in] &elementMatrix

    */
    void assemblyLaplacian(SmallMatrixPtr_Type &elementMatrix);

    /*!

     \brief Assembly advection vector field \f$ \int_T \nabla v \cdot u(\nabla u) ~dx\f$
    @param[in] &elementMatrix

    */
    void assemblyAdvection(SmallMatrixPtr_Type &elementMatrix);

    /*!
     \brief Assembly advection vector field in u
    @param[in] &elementMatrix

    */
    void assemblyAdvectionInU(SmallMatrixPtr_Type &elementMatrix);

    friend class AssembleFEFactory<SC, LO, GO, NO>; // Must have for specfic classes
    
    std::function<void(SC *, const SC *, const SC *)> featCallback_;
    std::vector<double> verts4feat_;
    std::vector<SC> featMat_;
    std::vector<SC> locConv_;

};

} // namespace FEDD
#endif
