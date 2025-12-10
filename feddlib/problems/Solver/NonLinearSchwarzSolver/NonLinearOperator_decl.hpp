#ifndef NONLINEAROPERATOR_DECL_hpp
#define NONLINEAROPERATOR_DECL_hpp

#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map.hpp"
#include <FROSch_SchwarzOperator_decl.hpp>

/*!
 Declaration of a nonlinear operator
 Simple interface that nonlinear operators implement, providing a non-const apply() function. Originally required in the
 NonLinearSumOperator which could not work with the stored array of Schwarz operators since these only have const
 apply() functions. This is a workaround to deal with FROSch_SchwarzOperator representing linear operators.

 @brief  Nonlinear Operator
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {
template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class NonLinearOperator {

  protected:
    // NOTE: [KH] 28.07.25 Using Tpetra MV here even though FROSch still uses Xpetra for now. Since the nonlinear
    // Schwarz ops are only being used by the FEDDLib at the moment and the FEDDLib has switched to Tpetra, this avoids
    // dynamic casting with toXpetra in many places.
    using TMultiVector = Tpetra::MultiVector<SC, LO, GO, NO>;
    using BlockMultiVectorPtrFEDD = typename Teuchos::RCP<FEDD::BlockMultiVector<SC, LO, GO, NO>>;
    using MapConstPtrFEDD = typename Teuchos::RCP<const FEDD::Map<LO, GO, NO>>;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    NonLinearOperator() = default;
    ~NonLinearOperator() = default;

    // non-const apply() interface which nonlinear operators should implement since they cannot be applied without
    // performing internal calculations. The input x is not const since overloads of the apply method convert Tpetra
    // MultiVectors to FEDD MultiVectors. If the input was const, a deep copy would have to be made. Now a shallow copy
    // is possible
    virtual void apply(TMultiVector &x, TMultiVector &y, SC alpha = ST::one(), SC beta = ST::zero()) = 0;
};
} // namespace FROSch
#endif
