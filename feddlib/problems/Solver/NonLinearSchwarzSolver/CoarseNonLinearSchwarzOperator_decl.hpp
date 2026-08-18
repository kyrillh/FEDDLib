#ifndef COARSENONLINEARSCHWARZOPERATOR_DECL_HPP
#define COARSENONLINEARSCHWARZOPERATOR_DECL_HPP

#include "feddlib/core/General/BCBuilder_decl.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/Mesh/Mesh_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp"
#include "feddlib/problems/abstract/NonLinearProblem_decl.hpp"
#include <FROSch_IPOUHarmonicCoarseOperator_decl.hpp>
#include <FROSch_SchwarzOperator_decl.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_MultiVector_decl.hpp>

/*!
 Declaration of CoarseNonLinearSchwarzOperator

 @brief Implements the coarse correction T_0 from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class CoarseNonLinearSchwarzOperator : public IPOUHarmonicCoarseOperator<SC, LO, GO, NO>,
                                       public NonLinearOperator<SC, LO, GO, NO> {

  protected:
    using ConstXMapPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtr;
    using ConstXMultiVectorPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMultiVectorPtr;
    using ParameterListPtr = typename SchwarzOperator<SC, LO, GO, NO>::ParameterListPtr;

    using UN = typename SchwarzOperator<SC, LO, GO, NO>::UN;
    using XMultiVector = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVector;
    using TMultiVector = typename NonLinearOperator<SC, LO, GO, NO>::TMultiVector;

    using NonLinearProblemPtrFEDD = typename Teuchos::RCP<FEDD::NonLinearProblem<SC, LO, GO, NO>>;
    using BlockMatrixPtrFEDD = typename Teuchos::RCP<FEDD::BlockMatrix<SC, LO, GO, NO>>;
    using BlockMultiVectorFEDD = typename FEDD::BlockMultiVector<SC, LO, GO, NO>;
    using BlockMultiVectorPtrFEDD = typename Teuchos::RCP<FEDD::BlockMultiVector<SC, LO, GO, NO>>;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    explicit CoarseNonLinearSchwarzOperator(NonLinearProblemPtrFEDD problem, ParameterListPtr parameterList);

    ~CoarseNonLinearSchwarzOperator() = default;

    int initialize() override;

    // the compute method is implemented in FROSch_CoarseOperator_def

    void apply(const BlockMultiVectorPtrFEDD x, BlockMultiVectorPtrFEDD y, SC alpha = ST::one(), SC beta = ST::zero());

    void apply(TMultiVector &x, TMultiVector &y, SC alpha = ST::one(), SC beta = ST::zero()) override;

    // This apply method must be overridden but does not make sense in the context of nonlinear operators
    void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode = NO_TRANS,
               SC alpha = ST::one(), SC beta = ST::zero()) const override;

    void exportCoarseBasis();

    int getRunStats() const;

    void describe(FancyOStream &out, const EVerbosityLevel verbLevel = Describable::verbLevel_default) const override;

    string description() const override;

  private:
    // FEDDLib problem object. (will need to be changed for interoperability)
    NonLinearProblemPtrFEDD problem_;
    // Current point of evaluation. Null if none has been passed
    BlockMultiVectorPtrFEDD x_;
    // Current output. Null if no valid output stored.
    BlockMultiVectorPtrFEDD y_;

    // Newtons method params
    double relNewtonTol_;
    double absNewtonTol_;
    int maxNumIts_;
    bool useBT_;
    // Newtons method helpers
    Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> coarseResidualVec_;
    Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> coarseDeltaG0_;
    Teuchos::RCP<FEDD::MultiVector<SC, LO, GO, NO>> deltaG0Merged_;
    BlockMultiVectorPtrFEDD deltaG0_;
    // Temp. problem state params
    BlockMultiVectorPtrFEDD solutionTmp_;
    BlockMatrixPtrFEDD systemTmp_;
    BlockMultiVectorPtrFEDD rhsTmp_;
    BlockMultiVectorPtrFEDD sourceTermTmp_;
    BlockMultiVectorPtrFEDD previousSolutionTmp_;
    BlockMultiVectorPtrFEDD residualVecTmp_;

    // Store total iteration count of inner Newton methods over all calls to apply()
    int totalIters_;
};

} // namespace FROSch

#endif
