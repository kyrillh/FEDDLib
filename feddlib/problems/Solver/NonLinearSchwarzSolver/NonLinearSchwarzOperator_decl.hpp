#ifndef NONLINEARSCHWARZOPERATOR_DECL_HPP
#define NONLINEARSCHWARZOPERATOR_DECL_HPP

#include "feddlib/core/FE/FE_decl.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/Mesh/Mesh_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp"
#include "feddlib/problems/abstract/NonLinearProblem_decl.hpp"
#include <FROSch_SchwarzOperator_decl.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <vector>

/*!
 Declaration of NonLinearSchwarzOperator

 @brief Implements the surrogate problem $\mathcal{F}(u)$ from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {
// TODO: these should be moved into the nonlinear Schwarz solver once created
enum class CombinationMode { Averaging, Full, Restricted };

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class NonLinearSchwarzOperator : public SchwarzOperator<SC, LO, GO, NO>, public NonLinearOperator<SC, LO, GO, NO> {

  protected:
    using CommPtr = typename SchwarzOperator<SC, LO, GO, NO>::CommPtr;

    using XMultiVector = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVector;
    using TMultiVector = typename NonLinearOperator<SC, LO, GO, NO>::TMultiVector;

    using ParameterListPtr = typename SchwarzOperator<SC, LO, GO, NO>::ParameterListPtr;

    using NonLinearProblemPtrFEDD = typename Teuchos::RCP<FEDD::NonLinearProblem<SC, LO, GO, NO>>;
    using BlockMatrixPtrFEDD = typename Teuchos::RCP<FEDD::BlockMatrix<SC, LO, GO, NO>>;
    using MatrixPtrFEDD = typename Teuchos::RCP<FEDD::Matrix<SC, LO, GO, NO>>;
    using BlockMultiVectorPtrFEDD = typename Teuchos::RCP<FEDD::BlockMultiVector<SC, LO, GO, NO>>;
    using MapConstPtrFEDD = typename Teuchos::RCP<const FEDD::Map<LO, GO, NO>>;
    using BlockMapPtrFEDD = typename Teuchos::RCP<FEDD::BlockMap<LO, GO, NO>>;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    struct RunStats {
        std::vector<int> totalIters; // Ordered by rank; nonempty only on rank 0
        std::vector<SC> totalTimes; // Ordered by rank; nonempty only on rank 0
        LO minIters = 0;
        SC avgIters = 0;
        LO maxIters = 0;
    };

    explicit NonLinearSchwarzOperator(CommPtr serialComm, NonLinearProblemPtrFEDD problem,
                                      ParameterListPtr parameterList);

    ~NonLinearSchwarzOperator() = default;

    int initialize() override;

    int compute() override;

    void apply(const BlockMultiVectorPtrFEDD x, BlockMultiVectorPtrFEDD y, SC alpha = ST::one(), SC beta = ST::zero());

    void apply(TMultiVector &x, TMultiVector &y, SC alpha = ST::one(), SC beta = ST::zero()) override;

    // This apply method must be overridden but does not make sense in the context of nonlinear operators
    void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode = NO_TRANS,
               SC alpha = ST::one(), SC beta = ST::zero()) const override;

    BlockMatrixPtrFEDD getLocalJacobianGhosts() const;

    RunStats getRunStats() const;

    void describe(FancyOStream &out, const EVerbosityLevel verbLevel = Describable::verbLevel_default) const override;

    string description() const override;

  private:
    void replaceMapAndExportProblem();

    // FEDDLib problem object. (will need to be changed for interoperability)
    NonLinearProblemPtrFEDD problem_;
    // Current point of evaluation. Null if none has been passed
    BlockMultiVectorPtrFEDD x_;
    // Current output. Null if no valid output stored.
    BlockMultiVectorPtrFEDD y_;
    // Tangent of the nonlinear problem R_iDF(u_i)P_i as used in ASPEN
    BlockMatrixPtrFEDD localJacobianGhosts_;
    // Local (serial) overlapping map object with one ghost layer
    BlockMapPtrFEDD blockElementMapLocal_;
    BlockMapPtrFEDD blockMapVecFieldOverlappingGhostsLocal_;
    BlockMapPtrFEDD blockMapOverlappingGhostsLocal_;
    // Sparsity pattern for NavierStokes class specifically. Nasty hack, but improves runtimes over calling
    // establishNNZPattern() repeatedly.
    MatrixPtrFEDD NNZ_A_;

    // Newtons method params
    SC relNewtonTol_;
    SC absNewtonTol_;
    int maxNumIts_;

    // Recombination mode. [Restricted, Averaging, Addition]
    CombinationMode combinationMode_;
    BlockMultiVectorPtrFEDD multiplicity_;

    // The vector "a" from the local pressure projections found e.g. in dissertation of Christian Hochmuth.
    // Analogue variable to OverlappingOperator->aProjection_.
    BlockMultiVectorPtrFEDD aProjection_;
    std::vector<SC> sumAA_;

    // Maps for saving the mpiComm maps of the problems domain when replacing them with serial maps
    BlockMapPtrFEDD blockElementMapMpiTmp_;
    BlockMapPtrFEDD blockMapRepeatedMpiTmp_;
    BlockMapPtrFEDD blockMapUniqueMpiTmp_;
    BlockMapPtrFEDD blockMapVecFieldRepeatedMpiTmp_;
    BlockMapPtrFEDD blockMapVecFieldUniqueMpiTmp_;

    // Vectors for saving repeated and unique points
    std::vector<FEDD::vec2D_dbl_ptr_Type> pointsRepTmp_;
    std::vector<FEDD::vec2D_dbl_ptr_Type> pointsUniTmp_;
    // Vectors for saving the boundary conditions
    std::vector<FEDD::vec_int_ptr_Type> bcFlagRepTmp_;
    std::vector<FEDD::vec_int_ptr_Type> bcFlagUniTmp_;
    // Vector of elements for saving elementsC_
    std::vector<Teuchos::RCP<FEDD::Elements>> elementsCTmp_;
    // Current global solution of the problem
    BlockMatrixPtrFEDD systemTmp_;
    MatrixPtrFEDD NNZ_A_Tmp_;
    BlockMultiVectorPtrFEDD solutionTmp_;
    BlockMultiVectorPtrFEDD rhsTmp_;
    BlockMultiVectorPtrFEDD sourceTermTmp_;
    BlockMultiVectorPtrFEDD previousSolutionTmp_;
    BlockMultiVectorPtrFEDD residualVecTmp_;
    // FE assembly factory for global and local assembly
    Teuchos::RCP<FEDD::FE<SC, LO, GO, NO>> feFactoryTmp_;
    Teuchos::RCP<FEDD::FE<SC, LO, GO, NO>> feFactoryGhostsLocal_;
    // Store total iteration count of inner Newton methods over all calls to apply()
    int totalIters_;
};

} // namespace FROSch

#endif
