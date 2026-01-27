#ifndef SIMPLEOVERLAPPINGOPERATPR_DEF_HPP
#define SIMPLEOVERLAPPINGOPERATPR_DEF_HPP

#include "SimpleOverlappingOperator_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include "feddlib/core/FE/Domain_decl.hpp"
#include <FROSch_OverlappingOperator_decl.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_EReductionType.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_implicit_cast.hpp>
#include <Xpetra_ConfigDefs.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_ExportFactory.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_Map_decl.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MultiVectorFactory_decl.hpp>
#include <stdexcept>
#include <string>
#include <trilinos_amd.h>

namespace FROSch {

template <class SC, class LO, class GO, class NO>
SimpleOverlappingOperator<SC, LO, GO, NO>::SimpleOverlappingOperator(NonLinearProblemPtrFEDD problem,
                                                                     ParameterListPtr parameterList)
    : OverlappingOperator<SC, LO, GO, NO>(
          FEDD::toXpetraMatrix(problem->system_->getMergedMatrix()->getTpetraMatrixNonConst()), parameterList),
      uniqueMap_(), importerUniqueToGhosts_(), x_Ghosts_(), y_unique_(), y_Ghosts_(), bcFlagOverlappingGhostsVec_(),
      problem_(problem) {
    // Override the combine mode of the FROSch operator base object from the nonlinear Schwarz configuration
    auto combineModeTemp = problem_->getParameterList()->get("Combine Mode", "Restricted");
    if (combineModeTemp == "Averaging") {
        this->Combine_ = OverlappingOperator<SC, LO, GO, NO>::CombinationType::Averaging;
    } else if (combineModeTemp == "Full") {
        this->Combine_ = OverlappingOperator<SC, LO, GO, NO>::CombinationType::Full;
    } else if (combineModeTemp == "Restricted") {
        this->Combine_ = OverlappingOperator<SC, LO, GO, NO>::CombinationType::Restricted;
    } else {
        if (this->MpiComm_->getRank() == 0) {
            std::cerr << "\nInvalid Recombination Mode in SimpleOverlappingOperator: \"" << combineModeTemp
                      << "\". Defaulting to \"Restricted\"" << std::endl;
        }
        this->Combine_ = OverlappingOperator<SC, LO, GO, NO>::CombinationType::Restricted;
    }
}

template <class SC, class LO, class GO, class NO>
int SimpleOverlappingOperator<SC, LO, GO, NO>::initialize(
    CommPtr serialComm, ConstXMatrixPtr jacobianGhosts, ConstXMapPtr overlappingMap, ConstXMapPtr overlappingGhostsMap,
    ConstXMapPtr uniqueMap, std::vector<FEDD::vec_int_ptr_Type> bcFlagOverlappingGhostsVec) {
    FROSCH_TIMER_START(SimpleOverlappingInitialize, " SimpleOverlapping::initialize");
    // AlgebraicOverlappingOperator does: calculates overlap multiplicity if needed and does symbolic extraction
    // of local subdomain matrix and initialization of solver (symbolic factorization)
    // Here we just read in the localSubdomainMatrix since it already exists

    // Need to set the following as AlgebraicOverlappingOperator initialize()
    // x OverlappingMap_ (global)
    // x OverlappingMatrix_ (global and local)
    // x Multiplicity_
    // x SubdomainSolver_
    // x IsInitialzed_
    // x IsComputed_
    // - subdomainMatrix_ (NO: only used as an intermediate overlapping matrix from which values are taken for
    // localSubdomainMatrix_)
    // - localSubdomainMatrix_ (NO: this is the non-const predecessor of OverlappingMatrix_ used for extraction and
    // factorization in AlgebraicOverlappingOperator)
    // - ExtractLocalSubdomainMatrix_Symbolic_Done flag (NO: should not need this since not doing any symbolic
    // extraction)

    // Need to pass the serial communicator on which localJacobian lives
    this->SerialComm_ = serialComm;
    this->OverlappingMatrix_ = jacobianGhosts;
    uniqueMap_ = uniqueMap;
    bcFlagOverlappingGhostsVec_ = bcFlagOverlappingGhostsVec;

    // Initialize importer Ghosts -> Ghosts
    importerUniqueToGhosts_ = Xpetra::ImportFactory<LO, GO>::Build(uniqueMap, overlappingGhostsMap);

    //  Calculate overlap multiplicity if needed. OverlappingMap without ghosts required for this
    this->OverlappingMap_ = overlappingMap;
    this->initializeOverlappingOperator();
    // Distributed map corresponding to OverlappingMatrix_
    this->OverlappingMap_ = overlappingGhostsMap;
    // Need to rebuild Scatter_ to use the correct overlapping map
    this->Scatter_ = ImportFactory<LO,GO,NO>::Build(this->getDomainMap(),this->OverlappingMap_);

    // Compute symbolic factorization
    this->initializeSubdomainSolver(this->OverlappingMatrix_);
    this->IsInitialized_ = true;
    this->IsComputed_ = false;
    return 0;
}

template <class SC, class LO, class GO, class NO> int SimpleOverlappingOperator<SC, LO, GO, NO>::compute() {
    FROSCH_TIMER_START(SimpleOverlappingInitialize, " SimpleOverlapping::compute");
    // AlgebraicOverlappingOperator does: gets values of the local subdomain matrices and computes the numerical
    // factorization
    // Here we do not need to fill values into the sparsity pattern since they are already there
    // ==> updateLocalOverlappingMatrices() is a no-op
    TEUCHOS_TEST_FOR_EXCEPTION(!this->IsInitialized_, std::runtime_error,
                               "ASPENOverlappingOperator must be initialized before calling compute()");
    this->computeOverlappingOperator();
    return 0;
}

// NOTE: FROSch_OverlappingOperator->apply() expects a uniquely distributed input and applies the global matrix K to the
// global input before applying the restriction operator R_i. This prevents using FROSch_OverlappingOperator for
// evaluation of D\mathcal{F}(u) since the input must first be communicated to an overlapping vector so that R_iDF(u_i)
// can be applied on each subdomain. Instead we need to use this modified version.
template <class SC, class LO, class GO, class NO>
void SimpleOverlappingOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, ETransp mode, SC alpha,
                                                      SC beta) const {

    FEDD_TIMER_START(DFuTimer, " - Schwarz - apply DFu");
    // y = alpha*f(x) + beta*y
    // move the input to the local serial overlapping ghosts map
    // this->OverlappingMap_ partitions the subdomains with overlap and ghost layer and uses the global comm
    // this->OverlappingMatrix_->rowMap_ has the same partitioning but uses a serial comm
    if (x_Ghosts_.is_null()) {
        x_Ghosts_ = Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(this->OverlappingMap_, x.getNumVectors());
    } else {
        x_Ghosts_->putScalar(0.);
        x_Ghosts_->replaceMap(this->OverlappingMap_);
    }
    FEDD_TIMER_START(ImportTimer, " - Schwarz - import x");
    x_Ghosts_->doImport(x, *importerUniqueToGhosts_, Xpetra::CombineMode::ADD);
    FEDD_TIMER_STOP(ImportTimer);
    x_Ghosts_->replaceMap(this->OverlappingMatrix_->getRowMap());

    //  Apply DF(u_i)
    this->OverlappingMatrix_->apply(*x_Ghosts_, *x_Ghosts_, mode, ST::one(), ST::zero());
    // Set solution on ghost points to zero so that column entries in (R_iDF(u_i)P_i)^-1 corresponding to ghost nodes do
    // not affect the solution. This is equivalent to applying the restriction operator R_i. This should also result in
    // ghost entries that are zero after the solve since rows in R_iDF(u_i)P_i corresponding to ghost nodes were set to
    // zero during inner Newton solves in NonLinearSchwarzOperator.
    int blockOffset = 0;
    for (int i = 0; i < bcFlagOverlappingGhostsVec_.size(); i++) {
        auto numNodesBlock = bcFlagOverlappingGhostsVec_.at(i)->size();
        auto dofsPerNode = problem_->getDofsPerNode(i);
        for (int j = 0; j < numNodesBlock; j++) {
            if (bcFlagOverlappingGhostsVec_.at(i)->at(j) == -99) {
                for (int k = 0; k < dofsPerNode; k++) {
                    x_Ghosts_->replaceLocalValue(blockOffset + dofsPerNode * j + k, 0, ST::zero());
                }
            }
        }
        blockOffset += numNodesBlock * dofsPerNode;
    }
    // Apply local solution
    if (y_Ghosts_.is_null()) {
        y_Ghosts_ =
            Xpetra::MultiVectorFactory<SC, LO, GO>::Build(this->OverlappingMatrix_->getRowMap(), x.getNumVectors());
    } else {
        y_Ghosts_->replaceMap(this->OverlappingMatrix_->getRowMap());
    }
    this->SubdomainSolver_->apply(*x_Ghosts_, *y_Ghosts_, mode, ST::one(), ST::zero());
    y_Ghosts_->replaceMap(this->OverlappingMap_);

    // Apply local pressure correction
    if (!this->aProjection_.is_null() &&
        (this->ParameterList_->sublist("Parameter").get("Use Pressure Projection", false) == true)) {

        FROSCH_TIMER_START_LEVELID(applyTime, "Apply Pressure Projection");

        RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout));
        // Here OverlappingMap_ represents the overlapping subdomain distribution including ghost layer on the global
        // comm
        XMultiVectorPtr a = MultiVectorFactory<SC, LO, GO, NO>::Build(this->OverlappingMap_, x.getNumVectors());
        auto importer = ImportFactory<LO, GO, NO>::Build(this->uniqueMap_, this->OverlappingMap_);

        // Get the projection on the overlapping subdomain
        // INSERT and ADD should be the same here since we are going from unique to overlapping
        //TODO:[KH] move this into init so it is only done once
        a->doImport(*this->aProjection_, *this->Scatter_, INSERT);
        blockOffset = 0;
        for (int i = 0; i < bcFlagOverlappingGhostsVec_.size(); i++) {
            auto numNodesBlock = bcFlagOverlappingGhostsVec_.at(i)->size();
            auto dofsPerNode = problem_->getDofsPerNode(i);
            for (int j = 0; j < numNodesBlock; j++) {
                if (bcFlagOverlappingGhostsVec_.at(i)->at(j) == -99) {
                    for (int k = 0; k < dofsPerNode; k++) {
                        a->replaceLocalValue(static_cast<LO>(blockOffset + dofsPerNode * j + k), 0, ST::zero());
                    }
                }
            }
            blockOffset += numNodesBlock * dofsPerNode;
        }

        // Perform local dot products
        SCVecPtr a_values = a->getDataNonConst(0);
        SCVecPtr y_values = this->y_Ghosts_->getDataNonConst(0);
        double sumAY = 0.;
        for (int i = 0; i < a_values.size(); i++) {
            sumAY += a_values[i] * y_values[i];
        }
        double sumAA = 0.;
        if (this->sumAA_ < 0.) {
            for (int i = 0; i < a_values.size(); i++) {
                sumAA += a_values[i] * a_values[i];
            }
            this->sumAA_ = sumAA;
        }
        double aint = 1. / this->sumAA_;
        SC scaling = aint * sumAY;
        y_Ghosts_->update(-scaling, *a, 1);
    }

    if (y_unique_.is_null()) {
        y_unique_ = Xpetra::MultiVectorFactory<SC, LO, GO>::Build(uniqueMap_, x.getNumVectors());
    } else {
        y_unique_->putScalar(ST::zero());
    }

    FEDD_TIMER_START(ExportTimer, " - Schwarz - export x");
    if (this->Combine_ == OverlappingOperator<SC, LO, GO, NO>::CombinationType::Restricted) {
        GO globalID = 0;
        LO localID = 0;
        for (auto i = 0; i < y_unique_->getNumVectors(); i++) {
            auto y_Ghosts_Data = y_Ghosts_->getData(i);
            for (auto j = 0; j < uniqueMap_->getLocalNumElements(); j++) {
                globalID = uniqueMap_->getGlobalElement(j);
                localID = this->OverlappingMap_->getLocalElement(globalID);
                y_unique_->getDataNonConst(i)[j] = y_Ghosts_Data[localID];
            }
        }
    } else {
        // Use export operation here since oldSolution is on overlapping map and newSolution on the unique map
        // Use Insert since newSolution does not contain any values yet
        FEDD_TIMER_START(ExportTimer, " - Schwarz - export x");
        y_unique_->doExport(*y_Ghosts_, *importerUniqueToGhosts_, ADD);
        FEDD_TIMER_STOP(ExportTimer);
    }
    FEDD_TIMER_START(AveragingTimer, " - Schwarz - average overlap");
    if (this->Combine_ == OverlappingOperator<SC, LO, GO, NO>::CombinationType::Averaging) {
        auto scaling = this->Multiplicity_->getData(0);
        for (auto i = 0; i < y_unique_->getNumVectors(); i++) {
            auto values = y_unique_->getDataNonConst(i);
            for (auto j = 0; j < values.size(); j++) {
                values[j] = values[j] / scaling[j];
            }
        }
    }
    FEDD_TIMER_STOP(AveragingTimer);
    y.update(alpha, *y_unique_, beta);
}

// The standard way to apply this operator is with the local Jacobians --> usePreonditionerOnly is not required for
// this.
template <class SC, class LO, class GO, class NO>
void SimpleOverlappingOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y,
                                                      bool usePreconditionerOnly, ETransp mode, SC alpha,
                                                      SC beta) const {
    apply(x, y, mode, alpha, beta);
}

// TODO: a lot of this implementation could be moved to FROSch
template <class SC, class LO, class GO, class NO>
void SimpleOverlappingOperator<SC, LO, GO, NO>::describe(FancyOStream &out, const EVerbosityLevel verbLevel) const {
    using std::endl;
    using std::setw;
    using Teuchos::ArrayView;
    using Teuchos::Comm;
    using Teuchos::RCP;
    using Teuchos::TypeNameTraits;
    using Teuchos::VERB_DEFAULT;
    using Teuchos::VERB_EXTREME;
    using Teuchos::VERB_HIGH;
    using Teuchos::VERB_LOW;
    using Teuchos::VERB_MEDIUM;
    using Teuchos::VERB_NONE;

    const Teuchos::EVerbosityLevel vl = (verbLevel == VERB_DEFAULT) ? VERB_LOW : verbLevel;

    if (vl == VERB_NONE) {
        return; // Don't print anything at all
    }

    // By convention, describe() always begins with a tab.
    Teuchos::OSTab tab0(out);

    RCP<const Comm<int>> comm = this->MpiComm_;
    const int myRank = comm->getRank();
    const int numProcs = comm->getSize();
    size_t width = 1;
    for (size_t dec = 10; dec < this->OverlappingMatrix_->getGlobalNumRows(); dec *= 10) {
        ++width;
    }
    width = std::max<size_t>(width, static_cast<size_t>(11)) + 2;
    // set boolean printing to true/false
    cout << std::boolalpha;

    //    none: print nothing
    //     low: print O(1) info from node 0
    //  medium: print O(P) info, num entries per process
    //    high: print O(N) info, num entries per row
    // extreme: print O(NNZ) info: print indices and values
    //
    // for medium and higher, print constituent objects at specified verbLevel
    if (myRank == 0) {
        out << "Precomputed Overlapping Schwarz Operator:" << endl;
    }
    Teuchos::OSTab tab1(out);

    if (myRank == 0) {
        if (this->getObjectLabel() != "") {
            out << "Label: \"" << this->getObjectLabel() << "\", ";
        }
        {
            out << "Template parameters:" << endl;
            Teuchos::OSTab tab2(out);
            out << "Scalar: " << TypeNameTraits<SC>::name() << endl
                << "LocalOrdinal: " << TypeNameTraits<LO>::name() << endl
                << "GlobalOrdinal: " << TypeNameTraits<GO>::name() << endl
                << "Node: " << TypeNameTraits<NO>::name() << endl;
        }
        if (this->OverlappingMatrix_->isFillComplete()) {
            out << "isFillComplete: true" << endl
                << "Global dimensions: [" << this->OverlappingMatrix_->getGlobalNumRows() << ", "
                << this->OverlappingMatrix_->getGlobalNumCols() << "]" << endl
                << "Global number of entries: " << this->OverlappingMatrix_->getGlobalNumEntries() << endl
                << endl
                << "Global max number of entries in a row: " << this->OverlappingMatrix_->getGlobalMaxNumRowEntries()
                << endl;
        } else {
            out << "isFillComplete: false" << endl
                << "Global dimensions: [" << this->OverlappingMatrix_->getGlobalNumRows() << ", "
                << this->OverlappingMatrix_->getGlobalNumCols() << "]" << endl;
        }
        out << "isInitialized: " << this->IsInitialized_ << endl;
        out << "isComputed: " << this->IsComputed_ << endl;
    }

    if (vl < VERB_MEDIUM) {
        return; // all done!
    }

    // Describe the row Map.
    if (myRank == 0) {
        out << endl << "Row Map:" << endl;
    }
    if (this->OverlappingMatrix_->getRowMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->OverlappingMatrix_->getRowMap()->describe(out, vl);
    }

    // Describe the column Map.
    if (myRank == 0) {
        out << "Column Map: ";
    }
    if (this->OverlappingMatrix_->getColMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else if (this->OverlappingMatrix_->getColMap() == this->OverlappingMatrix_->getRowMap()) {
        if (myRank == 0) {
            out << "same as row Map" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->OverlappingMatrix_->getColMap()->describe(out, vl);
    }

    // Describe the domain Map.
    if (myRank == 0) {
        out << "Domain Map: ";
    }
    if (this->OverlappingMatrix_->getDomainMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else if (this->OverlappingMatrix_->getDomainMap() == this->OverlappingMatrix_->getRowMap()) {
        if (myRank == 0) {
            out << "same as row Map" << endl;
        }
    } else if (this->OverlappingMatrix_->getDomainMap() == this->OverlappingMatrix_->getColMap()) {
        if (myRank == 0) {
            out << "same as column Map" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->OverlappingMatrix_->getDomainMap()->describe(out, vl);
    }

    // Describe the range Map.
    if (myRank == 0) {
        out << "Range Map: ";
    }
    if (this->OverlappingMatrix_->getRangeMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else if (this->OverlappingMatrix_->getRangeMap() == this->OverlappingMatrix_->getDomainMap()) {
        if (myRank == 0) {
            out << "same as domain Map" << endl;
        }
    } else if (this->OverlappingMatrix_->getRangeMap() == this->OverlappingMatrix_->getRowMap()) {
        if (myRank == 0) {
            out << "same as row Map" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->OverlappingMatrix_->getRangeMap()->describe(out, vl);
    }

    if (vl < VERB_HIGH) {
        return; // all done!
    }

    // O(N) and O(NNZ) data
    for (int curRank = 0; curRank < numProcs; ++curRank) {
        if (myRank == curRank) {
            out << std::setw(width) << "Proc Rank" << std::setw(width) << "Global Row" << std::setw(width)
                << "Num Entries";
            if (vl == VERB_EXTREME) {
                out << std::setw(width) << "(Index,Value)";
            }
            out << endl;
            for (size_t r = 0; r < this->OverlappingMatrix_->getLocalNumRows(); ++r) {
                const size_t nE = this->OverlappingMatrix_->getNumEntriesInLocalRow(r);
                GO gid = this->OverlappingMatrix_->getRowMap()->getGlobalElement(r);
                out << std::setw(width) << myRank << std::setw(width) << gid << std::setw(width) << nE;
                if (vl == VERB_EXTREME) {
                    if (this->OverlappingMatrix_->isGloballyIndexed()) {
                        ArrayView<const GO> rowinds;
                        ArrayView<const SC> rowvals;
                        this->OverlappingMatrix_->getGlobalRowView(gid, rowinds, rowvals);
                        for (size_t j = 0; j < nE; ++j) {
                            out << " (" << rowinds[j] << ", " << rowvals[j] << ") ";
                        }
                    } else if (this->OverlappingMatrix_->isLocallyIndexed()) {
                        ArrayView<const LO> rowinds;
                        ArrayView<const SC> rowvals;
                        this->OverlappingMatrix_->getLocalRowView(r, rowinds, rowvals);
                        for (size_t j = 0; j < nE; ++j) {
                            out << " (" << this->OverlappingMatrix_->getColMap()->getGlobalElement(rowinds[j]) << ", "
                                << rowvals[j] << ") ";
                        }
                    } // globally or locally indexed
                } // vl == VERB_EXTREME
                out << endl;
            } // for each row r on this process
        } // if (myRank == curRank)

        // Give output time to complete
        comm->barrier();
        comm->barrier();
        comm->barrier();
    } // for each process p
}

template <class SC, class LO, class GO, class NO>
std::string SimpleOverlappingOperator<SC, LO, GO, NO>::description() const {
    return "ASPEN Overlapping Operator";
}

} // namespace FROSch

#endif // SimpleOVERLAPPINGOPERATPR_DEF_HPP
