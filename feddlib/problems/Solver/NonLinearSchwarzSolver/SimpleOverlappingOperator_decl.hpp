#ifndef SIMPLEOVERLAPPINGOPERATOR_DECL_HPP
#define SIMPLEOVERLAPPINGOPERATOR_DECL_HPP

#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/Mesh/Mesh_decl.hpp"
#include "feddlib/problems/abstract/NonLinearProblem_decl.hpp"
#include <Amesos2_Solver.hpp>
#include <FROSch_OverlappingOperator_decl.hpp>
#include <FROSch_SchwarzOperator_decl.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <stdexcept>
#include <vector>

/*!
 Declaration of ASPENOverlappingOperator

 @brief Implements the ASPEN tangent $D\mathcal{F}(u) = \sum P_i(R_iDF(u_i)P_i)^{-1}R_iDF(u_i)$ from the nonlinear
 Schwarz approach. FROSch::OverlappingOperator cannot be used directly because it uses a single matrix e.g. DF(u), but
 the ASPEN tangent consists of shifted matrices DF(u_i) evaluated on each subdomain. Furthermore, some FEDDLib::Problem
 specific code has made its way into this operator to efficiently deal with matrices on overlapping subdomains including
 ghost nodes.
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class SimpleOverlappingOperator : public OverlappingOperator<SC, LO, GO, NO> {

  protected:
    using CommPtr = typename SchwarzOperator<SC, LO, GO, NO>::CommPtr;

    using ConstXMapPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtr;

    using ConstXMatrixPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMatrixPtr;

    using XMultiVector = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVector;
    using XMultiVectorPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVectorPtr;
    using XMultiVectorConstPtr = const XMultiVectorPtr;

    using XImportPtr = typename SchwarzOperator<SC, LO, GO, NO>::XImportPtr;

    using ParameterListPtr = typename SchwarzOperator<SC, LO, GO, NO>::ParameterListPtr;

    using SCVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::SCVecPtr;

    using NonLinearProblemPtrFEDD = typename Teuchos::RCP<FEDD::NonLinearProblem<SC, LO, GO, NO>>;
    using MapConstPtrFEDD = typename Teuchos::RCP<const FEDD::Map<LO, GO, NO>>;
    using ST = typename Teuchos::ScalarTraits<SC>;
    using Amesos2SolverPtr =
        Teuchos::RCP<Amesos2::Solver<Tpetra::CrsMatrix<SC, LO, GO, NO>, Tpetra::MultiVector<SC, LO, GO, NO>>>;

  public:
    SimpleOverlappingOperator(NonLinearProblemPtrFEDD problem, ParameterListPtr parameterList);

    ~SimpleOverlappingOperator() = default;

    int initialize() override {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                   "SimpleOverlappingOperator requires local and global maps and serial "
                                   "and mpi comms during initialization");
    };

    /**
     * \brief initialization routine for SimpleOverlappingOperator. Since SimpleOverlappingOperator should use an
     * existing symbolically factorized solver object passed in using updateMatrixAndSolver(), this routine does not
     * call solver->initialize() like other FROSch operators.
     *
     * \param serialComm Serial MPI communicator
     * \param overlappingMap Map of the overlapping subdomain without ghost nodes
     * \param overlappingGhostsMap Map of the overlapping subdomain including ghost nodes
     * \param uniqueMap Map of the subdomain in a unique distribution of nodes
     */
    int initialize(CommPtr serialComm, ConstXMapPtr overlappingMap, ConstXMapPtr overlappingGhostsMap,
                   ConstXMapPtr uniqueMap);

    /**
     * This version of initialize is not used anymore. However it is kept for potential future uses if solvers other
     * than an amesos2 direct solver is used for local linear problems
     */
    int initialize(CommPtr serialComm, ConstXMatrixPtr jacobianGhosts, ConstXMapPtr overlappingMap,
                   ConstXMapPtr overlappingGhostsMap, ConstXMapPtr uniqueMap);

    /**
     * \brief Pass a new local matrix i.e. on a serial communicator, and symbolically factorized direct solver to
     * SimpleOverlappingOperator. The symbolically factorized direct solver only really needs to be passed once, but for
     * simplicity we pass it in every time here with minimal overhead.
     *
     * \param jacobianGhosts An Xpetra::matrix object on a serial communicator that lives on the local subdomain
     * including overlap and ghost layer.
     * \param amesos2SolverTpetra A Tpetra amesos2 solver interface object with an already symbolically factorized
     * solver stemming from the same discretization as jacobianGhosts i.e. the sparsity pattern of jacobianGhosts must
     * match that expected by the solver object.
     */
    int updateMatrixAndSolver(ConstXMatrixPtr jacobianGhosts, Amesos2SolverPtr amesos2SolverTpetra);

    int compute() override;

    void apply(const XMultiVector &x, XMultiVector &y, ETransp mode = NO_TRANS, SC alpha = ST::one(),
               SC beta = ST::zero()) const override;

    void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode = NO_TRANS,
               SC alpha = ST::one(), SC beta = ST::zero()) const override;

    void describe(FancyOStream &out, const EVerbosityLevel verbLevel = Describable::verbLevel_default) const override;

    string description() const override;

  protected:
    // Do nothing op in this case since the local overlapping matrices are already known
    int updateLocalOverlappingMatrices() override { return 0; }

  private:
    // Tangent of the nonlinear Schwarz operator is saved in this->OverlappingMatrix_ which lives on
    // serial version of OverlappingMap_
    // This operator does not know whether the system it is being applied to is a block system or not. This information
    // is only necessary in operators that do assembly. The following maps are intitialized to the correct dof maps

    // Distributed maps
    // GhostsMap is stored in this->OverlappingMap_
    ConstXMapPtr uniqueMap_;
    // Importers
    XImportPtr importerUniqueToGhosts_;

    // Temp. vectors for local results
    mutable XMultiVectorPtr x_Ghosts_;
    mutable XMultiVectorPtr y_unique_;
    mutable XMultiVectorPtr y_Ghosts_;

    // Boundary condition flags for recognizing the ghost boundary
    std::vector<FEDD::vec_int_ptr_Type> bcFlagOverlappingGhostsVec_;

    // We need to know how many dofs per node there are. This is stored in the problem object. We keep a pointer to this
    // object here to avoid having to copy the vector around.
    NonLinearProblemPtrFEDD problem_;
    // Recombination mode. [Restricted, Averaging, Addition]
    /* RecombinationMode recombinationMode_; */
    /* BlockMultiVectorPtrFEDD multiplicity_; */
};

} // namespace FROSch

#endif
