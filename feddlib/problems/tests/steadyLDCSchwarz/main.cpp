#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/General/ExporterParaView.hpp"
#include "feddlib/core/General/BCBuilder.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/core/Mesh/MeshPartitioner.hpp"
#include "feddlib/problems/Solver/NonLinearSolver.hpp"
#include "feddlib/problems/specific/NavierStokesAssFE.hpp"

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_TestForException.hpp>
#include <stdexcept>

void initialValue(double *x, double *res, double *parameters) {
    int dofs = static_cast<int>(parameters[0]);
    for (int i = 0; i < dofs; i++) {
        res[i] = 0;
    }
}

// Required for setting the Dirichlet BC on the ghost points to the current global solution in nonlinear Schwarz
void currentSolutionDirichlet3D(double *x, double *res, double t, const double *parameters) {
    res[0] = x[0];
    res[1] = x[1];
    res[2] = x[2];
}

void currentSolutionDirichlet2D(double *x, double *res, double t, const double *parameters) {
    res[0] = x[0];
    res[1] = x[1];
}
void currentSolutionDirichlet1D(double *x, double *res, double t, const double *parameters) { res[0] = x[0]; }

void zeroDirichlet(double *x, double *res, double t, const double *parameters) {
    res[0] = 0.;
    return;
}

void zeroDirichlet2D(double *x, double *res, double t, const double *parameters) {
    res[0] = 0.;
    res[1] = 0.;
    return;
}

void zeroDirichlet3D(double *x, double *res, double t, const double *parameters) {
    res[0] = 0.;
    res[1] = 0.;
    res[2] = 0.;
    return;
}

// For Lid Driven Cavity Test
void ldcFunc2D(double *x, double *res, double t, const double *parameters) {
    res[0] = 1.; //* parameters[0];
    res[1] = 0.;
    return;
}

// For Lid Driven Cavity Test
void ldcFunc3D(double *x, double *res, double t, const double *parameters) {
    res[0] = 1. * parameters[0];
    res[1] = 0.;
    res[2] = 0.;
    return;
}

void dummyFunc(double *x, double *res, double *parameters) {
    if (static_cast<int>(parameters[0]) == 2)
        res[0] = 1;
    else
        res[0] = 0.;
    return;
}

typedef unsigned UN;
typedef default_sc SC;
typedef default_lo LO;
typedef default_go GO;
typedef default_no NO;

using namespace FEDD;

int main(int argc, char *argv[]) {

    typedef MeshPartitioner<SC, LO, GO, NO> MeshPartitioner_Type;
    typedef Teuchos::RCP<Domain<SC, LO, GO, NO>> DomainPtr_Type;

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    Teuchos::RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    bool verbose(comm->getRank() == 0);

    // Command Line Parameters
    Teuchos::CommandLineProcessor myCLP;

    std::string xmlProblemFile = "parametersProblem.xml";
    myCLP.setOption("problemfile", &xmlProblemFile, ".xml file with Inputparameters.");
    std::string xmlSchwarzSolverFile = "parametersSolverNonLinSchwarz.xml";
    myCLP.setOption("schwarzsolverfile", &xmlSchwarzSolverFile, ".xml file with Inputparameters.");
    double length = 4.;
    myCLP.setOption("length", &length, "length of domain.");
    bool debug = false;
    myCLP.setOption("debug", "", &debug, "Bool option for debugging");

    myCLP.recogniseAllOptions(true);
    myCLP.throwExceptions(false);
    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = myCLP.parse(argc, argv);
    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
        MPI_Finalize();
        return 0;
    }

    if (comm->getRank() == 0 && debug) {
        waitForGdbAttach<LO>();
    }
    comm->barrier();
    comm->barrier();

    Teuchos::RCP<Teuchos::StackedTimer> stackedTimer = rcp(new Teuchos::StackedTimer("Nonlinear Schwarz solver", true));
    Teuchos::TimeMonitor::setStackedTimer(stackedTimer);

    ParameterListPtr_Type parameterListProblem = Teuchos::getParametersFromXmlFile(xmlProblemFile);

    ParameterListPtr_Type parameterListSolver = Teuchos::getParametersFromXmlFile(xmlSchwarzSolverFile);

    int dim = parameterListProblem->sublist("Parameter").get("Dimension", 3);

    std::string discVelocity = parameterListProblem->sublist("Parameter").get("Discretization Velocity", "P2");
    std::string discPressure = parameterListProblem->sublist("Parameter").get("Discretization Pressure", "P1");

    int m = parameterListProblem->sublist("Parameter").get("H/h", 5);
    auto overlap = parameterListSolver->get("Overlap", 1);
    int n;

    ParameterListPtr_Type parameterListAll(new Teuchos::ParameterList(*parameterListProblem));
    parameterListAll->setParameters(*parameterListSolver);

    int minNumberSubdomains = 1;

    int numProcsCoarseSolve = 0;
    int size = comm->getSize();

    DomainPtr_Type domainPressure;
    DomainPtr_Type domainVelocity;
    if (verbose) {
        std::cout << "-- Building Mesh ..." << std::flush;
    }

    MeshPartitioner_Type::DomainPtrArray_Type domainArray(2);
    domainArray[0] = domainPressure;
    domainArray[1] = domainVelocity;

    ParameterListPtr_Type pListPartitioner = sublist(parameterListAll, "Mesh Partitioner");
    MeshPartitioner<SC, LO, GO, NO> partitioner;

    // Structured Mesh for Lid-Driven Cavity Test
    TEUCHOS_TEST_FOR_EXCEPTION(size % minNumberSubdomains != 0, std::logic_error,
                               "Wrong number of processors for structured mesh.")
    if (dim == 2) {
        n = (int)(std::pow(size / minNumberSubdomains, 1 / 2.) + 100 * Teuchos::ScalarTraits<double>::eps()); // 1/H
        std::vector<double> x(2);
        x[0] = 0.0;
        x[1] = 0.0;
        domainPressure.reset(new Domain<SC, LO, GO, NO>(x, 1., 1., comm));
        domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, 1., 1., comm));
    } else if (dim == 3) {
        n = (int)(std::pow(size / minNumberSubdomains, 1 / 3.) + 100 * Teuchos::ScalarTraits<double>::eps()); // 1/H
        std::vector<double> x(3);
        x[0] = 0.0;
        x[1] = 0.0;
        x[2] = 0.0;
        domainPressure.reset(new Domain<SC, LO, GO, NO>(x, 1., 1., 1., comm));
        domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, 1., 1., 1., comm));
    }
    domainPressure->buildMesh(5, "Square", dim, discPressure, n, m, numProcsCoarseSolve);
    domainVelocity->buildMesh(5, "Square", dim, discVelocity, n, m, numProcsCoarseSolve);

    domainArray[0] = domainPressure;
    domainArray[1] = domainVelocity;
    // The FE type passed here is not used. The MeshPartitioner gets the FE type directly from each domain
    partitioner = MeshPartitioner<SC, LO, GO, NO>(domainArray, pListPartitioner, "P1", dim);
    // partitionerP1.tempSchwarzInitFunc(0);
    // partitionerP1.tempSchwarzInitFunc(1);
    partitioner.buildOverlappingDualGraphFromDistributedParMETIS(0, overlap);
    partitioner.buildOverlappingDualGraphFromDistributedParMETIS(1, overlap);

    partitioner.buildSubdomainFromDualGraphStructured(0);
    partitioner.buildSubdomainFromDualGraphStructured(1);

    std::vector<double> parameter_vec(1, parameterListProblem->sublist("Parameter").get("MaxVelocity", 1.));

    Teuchos::RCP<BCBuilder<SC, LO, GO, NO>> bcFactory(new BCBuilder<SC, LO, GO, NO>());

    parameter_vec.push_back(0.); // Dummy
    if (dim == 2) {
        bcFactory->addBC(zeroDirichlet2D, 1, 0, domainVelocity, "Dirichlet", dim);
        bcFactory->addBC(zeroDirichlet2D, 3, 0, domainVelocity, "Dirichlet", dim);
        bcFactory->addBC(ldcFunc2D, 2, 0, domainVelocity, "Dirichlet", dim, parameter_vec);
        bcFactory->addBC(zeroDirichlet, 3, 1, domainPressure, "Dirichlet", 1);
        // The current global solution must be set as the Dirichlet BC on the ghost nodes for nonlinear Schwarz solver
        // to correctly solve on the subdomains
        bcFactory->addBC(currentSolutionDirichlet2D, -99, 0, domainVelocity, "Dirichlet", dim);
    } else if (dim == 3) {
        bcFactory->addBC(zeroDirichlet3D, 1, 0, domainVelocity, "Dirichlet", dim);
        bcFactory->addBC(ldcFunc3D, 2, 0, domainVelocity, "Dirichlet", dim, parameter_vec);
        bcFactory->addBC(zeroDirichlet, 3, 1, domainPressure, "Dirichlet", 1); // Pressure Node
        bcFactory->addBC(currentSolutionDirichlet3D, -99, 0, domainVelocity, "Dirichlet", dim);
    }
    bcFactory->addBC(currentSolutionDirichlet1D, -99, 1, domainPressure, "Dirichlet", 1);

    NavierStokesAssFE<SC, LO, GO, NO> navierStokes(domainVelocity, discVelocity, domainPressure, discPressure,
                                                   parameterListAll);

    domainVelocity->info();
    domainPressure->info();
    navierStokes.info();

    navierStokes.addBoundaries(bcFactory);
    navierStokes.addRhsFunction(dummyFunc);

    navierStokes.initializeProblem();
    if (dim == 2) {
        navierStokes.initSolutionWithFunction(initialValue, 0, std::vector<double>{2});
        navierStokes.initSolutionWithFunction(initialValue, 1, std::vector<double>{1});
    } else if (dim == 3) {
        navierStokes.initSolutionWithFunction(initialValue, 0, std::vector<double>{3});
        navierStokes.initSolutionWithFunction(initialValue, 1, std::vector<double>{1});
    }
    navierStokes.assemble();
    navierStokes.setBoundariesRHS();

    std::string nlSolverType = parameterListProblem->sublist("General").get("Linearization", "NonlinearSchwarz");
    NonLinearSolver<SC, LO, GO, NO> nlSolver(nlSolverType);
    FEDD_TIMER_START(SolveTimer, " - Schwarz - global solve");
    nlSolver.solve(navierStokes);
    FEDD_TIMER_STOP(SolveTimer);
    comm->barrier();

    // Print a list of FEDD timers
    Teuchos::TimeMonitor::report(std::cout, "FEDD");
    stackedTimer->stop("Nonlinear Schwarz solver");
    Teuchos::StackedTimer::OutputOptions options;
    options.output_fraction = options.output_histogram = options.output_minmax = true;
    // Print the StackedTimer instance. This is a LIFO list of timers, typically constructed through
    // TimeMonitor::getNewCounter(). This function constructs non-existing timers and returns existing timers.
    // Note that TimeMonitor is a wrapper around timer objects. If a TimeMonitor object is constructed around a timer
    // object it starts the timer when it is constructed and stops the timer when it goes out of scope. Thus, repeated
    // calls to getNewCounter() wrapped in a TimeMonitor constructor can be used to time the cumulative runtime of a
    // function across multiple calls. This is taken advantage of e.g. by FEDD_TIMER_START.
    stackedTimer->report((std::cout), comm, options);

    if (parameterListAll->sublist("General").get("ParaViewExport", false)) {
        Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaVelocity(new ExporterParaView<SC, LO, GO, NO>());
        Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaPressure(new ExporterParaView<SC, LO, GO, NO>());

        Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionV = navierStokes.getSolution()->getBlock(0);
        Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionP = navierStokes.getSolution()->getBlock(1);

        DomainPtr_Type dom = domainVelocity;

        exParaVelocity->setup("velocity", dom->getMesh(), dom->getFEType());

        UN dofsPerNode = dim;
        exParaVelocity->addVariable(exportSolutionV, "u", "Vector", dofsPerNode, dom->getMapUnique());

        dom = domainPressure;
        exParaPressure->setup("pressure", dom->getMesh(), dom->getFEType());

        exParaPressure->addVariable(exportSolutionP, "p", "Scalar", 1, dom->getMapUnique());

        exParaVelocity->save(0.0);
        exParaPressure->save(0.0);
    }
    return (EXIT_SUCCESS);
}
