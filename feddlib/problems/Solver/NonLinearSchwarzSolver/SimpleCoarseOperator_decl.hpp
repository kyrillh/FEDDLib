#ifndef SIMPLECOARSEOPERATOR_DECL_HPP
#define SIMPLECOARSEOPERATOR_DECL_HPP

#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/Mesh/Mesh_decl.hpp"
#include "feddlib/problems/abstract/NonLinearProblem_decl.hpp"
#include <FROSch_CoarseOperator_decl.hpp>
#include <FROSch_SchwarzOperator_decl.hpp>
#include <FROSch_CoarseSpace_decl.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerbosityLevel.hpp>

/*!
 Declaration of simple coarse operator
 This class is just a wrapper around an existing NonlinearCoarseOperator providing it with an alternative apply() method
 that evaluates the tanget. The alternative apply() method is provided by the underlying CoarseOperator object. The
 initialize() method uses the default assignment operator of CoarseOperator to copy the member variables (which are
 mostly pointers) of the passed CoarseOperator object. By passing a NonlinearCoarseOperator object, its coarse Jacobian
 and coarse space can be used in the apply() method of CoarseOperator through the SimpleCoarseOperator object.

 CoarseOperator only implements an apply() method with the option usePreconditionerOnly. When calling this operator in a
 Thyra solver an apply() method without this option is required. This is provided via inheritance by the base class
 SchwarzOperator.

 @brief Implements the coarse tangent $D\mathcal{F}(u) = P_0(R_0DF(u_0)P_0)^{-1}R_0DF(u_0)$ from the nonlinear
 Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class SimpleCoarseOperator : public CoarseOperator<SC, LO, GO, NO> {

  protected:
    using ConstXMatrixPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMatrixPtr;

    using XMapPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMapPtr;
    using ConstXMapPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtr;
    using ConstXMapPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtrVecPtr;

    using ParameterListPtr = typename SchwarzOperator<SC, LO, GO, NO>::ParameterListPtr;

    using CoarseSpacePtr = typename SchwarzOperator<SC, LO, GO, NO>::CoarseSpacePtr;

    using UN = typename SchwarzOperator<SC, LO, GO, NO>::UN;

  public:
    SimpleCoarseOperator(ConstXMatrixPtr k, ParameterListPtr parameterList);

    ~SimpleCoarseOperator() = default;

    int initialize() override;
    int initialize(const Teuchos::RCP<const CoarseOperator<SC, LO, GO, NO>> inputOp);

    int compute() override;

    ConstXMapPtr computeCoarseSpace(CoarseSpacePtr coarseSpace) override;

    // These methods must be overriden, but are not used by the simple coarse operator
    int buildElementNodeList() override;
    int buildGlobalGraph(Teuchos::RCP<DDInterface<SC, LO, GO, NO>> theDDInterface_) override;
    XMapPtr BuildRepeatedMapCoarseLevel(ConstXMapPtr &nodesMap, UN dofsPerNode, ConstXMapPtrVecPtr dofsMaps,
                                        UN partition) override;

    int buildCoarseGraph() override;

    void describe(FancyOStream &out, const EVerbosityLevel verbLevel = Describable::verbLevel_default) const override;

    string description() const override;

  protected:
    void extractLocalSubdomainMatrix_Symbolic() override;

  private:
};

} // namespace FROSch

#endif
