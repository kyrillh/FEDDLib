#ifndef NONLINEARSUMOPERATOR_DECL_HPP
#define NONLINEARSUMOPERATOR_DECL_HPP
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearCombineOperator_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp"
#include <FROSch_SchwarzOperator_decl.hpp>
#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>

#include <FROSch_SumOperator_decl.hpp>
/*!
 Declaration of NonlinearSumOperator which extends the FROSch sum operator to allow non-const apply() methods. This is
 necessary since the apply() methods of nonlinear operators perform nonlinear solves which requires changing member
 variables.

 @brief Implements the coarse correction T_0 from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */
// NOTE: [KH] this class basically reimplements FROSch_SumOperator. If FROSch used virtual inheritance this operator
// could use most of the existing code since casting between NonlinearOperator and SchwarzOperator would be possible
// (without using dynamic_cast)

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class NonLinearSumOperator : public NonLinearCombineOperator<SC, LO, GO, NO> {

  protected:
    using CommPtr = typename NonLinearCombineOperator<SC, LO, GO, NO>::CommPtr;
    using TMultiVector = typename NonLinearCombineOperator<SC, LO, GO, NO>::TMultiVector;
    using UN = typename NonLinearCombineOperator<SC, LO, GO, NO>::UN;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    NonLinearSumOperator(CommPtr comm);

    ~NonLinearSumOperator() = default;

    void apply(TMultiVector &x, TMultiVector &y, SC alpha = ST::one(),
               SC beta = ST::zero()) override;
};
} // namespace FROSch

#endif
