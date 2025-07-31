#ifndef NONLINEARH1OPERATOR_DEF_HPP
#define NONLINEARH1OPERATOR_DEF_HPP

#include "NonLinearCombineOperator_decl.hpp"
#include "NonLinearH1Operator_decl.hpp"
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Tpetra_MultiVector_decl.hpp>
/*!
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC, class LO, class GO, class NO>
NonLinearH1Operator<SC, LO, GO, NO>::NonLinearH1Operator(CommPtr comm)
    : NonLinearCombineOperator<SC, LO, GO, NO>(comm), z0_{}, z1_{} {}

// Y = alpha * A^mode * X + beta * Y
template <class SC, class LO, class GO, class NO>
void NonLinearH1Operator<SC, LO, GO, NO>::apply(TMultiVector &x, TMultiVector &y, SC alpha, SC beta) {
    FROSCH_TIMER_START(NonLinearH1Apply, "NonLinearH1::apply");
    FROSCH_ASSERT(this->NonLinearOperatorVector_.size() == 2, "H1 operator can only be applied with two levels")

    // We do not explicitly check if the operators have been activated here as is done e.g. in the SumOperator
    // We can be sure that all operators are of type NonLinearOperator since this is checked when adding
    // We need to dynamic_cast here anyway because NonLinearOperator and SchwarzOperator are not related
    // This could be changed by modifying FROSch to allow virtual inheritance
    if (this->XTmpTpetra_.is_null())
        this->XTmpTpetra_ = Teuchos::rcp(new Tpetra::MultiVector<SC, LO, GO, NO>(x.getMap(), x.getNumVectors()));
    if (this->z1_.is_null())
        this->z1_ = Teuchos::rcp(new Tpetra::MultiVector<SC, LO, GO, NO>(y.getMap(), y.getNumVectors()));
    if (this->z0_.is_null())
        this->z0_ = Teuchos::rcp(new Tpetra::MultiVector<SC, LO, GO, NO>(y.getMap(), y.getNumVectors()));

    *this->XTmpTpetra_ = x;

    auto zero = ST::zero();
    auto one = ST::one();

    rcp_dynamic_cast<NonLinearOperator<SC, LO, GO, NO>>(this->NonLinearOperatorVector_[1])
        ->apply(*this->XTmpTpetra_, *this->z0_, one, zero);
    // Set YTmp_ = u - g0
    this->z1_->update(one, *this->XTmpTpetra_, zero);
    this->z1_->update(-one, *this->z0_, one);
    // Get overlapping correction g evaluated at u - g0
    // This will build the local Jacobians at v_i = u_0 - P_iT_i(u_0)
    rcp_dynamic_cast<NonLinearOperator<SC, LO, GO, NO>>(this->NonLinearOperatorVector_[0])
        ->apply(*this->z1_, *this->XTmpTpetra_, one, zero);
    // Add the coarse correction to the overlapping correction g(u-g0) + g0
    y.update(alpha, *this->z0_, beta);
    y.update(one, *this->XTmpTpetra_, one);
}

}; // namespace FROSch

#endif
