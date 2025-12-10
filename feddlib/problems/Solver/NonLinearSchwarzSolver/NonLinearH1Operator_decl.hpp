#ifndef NONLINEARH1OPERATOR_DECL_HPP
#define NONLINEARH1OPERATOR_DECL_HPP
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearCombineOperator_decl.hpp"
#include <FROSch_SchwarzOperator_decl.hpp>
#include <Teuchos_RCPDecl.hpp>

/*!
 Declaration of NonLinearH1Operator. This implements the nonlinear Schwarz operator in the hybrid-1 fashion as
introduced in "Additive and hybrid nonlinear two-level Schwarz methods and energy minimizing coarse spaces for
unstructured grids"

 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class NonLinearH1Operator : public NonLinearCombineOperator<SC, LO, GO, NO> {

  protected:
    using CommPtr = typename NonLinearCombineOperator<SC, LO, GO, NO>::CommPtr;
    using TMultiVector = typename NonLinearCombineOperator<SC, LO, GO, NO>::TMultiVector;
    using TMultiVectorPtr = typename Teuchos::RCP<TMultiVector>;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    NonLinearH1Operator(CommPtr comm);

    ~NonLinearH1Operator() = default;

    void apply(TMultiVector &x, TMultiVector &y, SC alpha = ST::one(),
               SC beta = ST::zero()) override;

  protected:
    mutable TMultiVectorPtr z0_;
    mutable TMultiVectorPtr z1_;
};
} // namespace FROSch

#endif
