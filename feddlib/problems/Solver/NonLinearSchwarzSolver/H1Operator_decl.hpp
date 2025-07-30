#ifndef H1OPERATOR_DECL_HPP
#define H1OPERATOR_DECL_HPP

#include "H1Operator_decl.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/CombineOperator_decl.hpp"
#include <FROSch_SchwarzOperator_def.hpp>

namespace FROSch {

/*!
 Declaration of H1Operator. This implements the Jacobian of the nonlinear Schwarz operator in the hybrid-1 fashion as introduced in
"Additive and hybrid nonlinear two-level Schwarz methods and energy minimizing coarse spaces for unstructured grids"

 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class H1Operator : public CombineOperator<SC, LO, GO, NO> {

  protected:
    using CommPtr = typename SchwarzOperator<SC, LO, GO, NO>::CommPtr;

    using XMultiVector = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVector;
    using XMultiVectorPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVectorPtr;

    using SchwarzOperatorPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtrVecPtr;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    H1Operator(CommPtr comm);

    H1Operator(SchwarzOperatorPtrVecPtr operators);

    ~H1Operator() = default;

    virtual void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode = NO_TRANS,
                       SC alpha = ST::one(), SC beta = ST::zero()) const override;

  protected:
    mutable XMultiVectorPtr z0_;
    mutable XMultiVectorPtr z1_;
};

} // namespace FROSch

#endif
