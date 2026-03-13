#ifndef NONLINEARSOLVER_DEF_hpp
#define NONLINEARSOLVER_DEF_hpp
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/CoarseNonLinearSchwarzOperator.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/H1Operator.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearH1Operator.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearSchwarzOperator.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/SumOperator.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearSumOperator.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/SimpleCoarseOperator.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/SimpleOverlappingOperator.hpp"
#include "feddlib/problems/Solver/Preconditioner.hpp"
#include "feddlib/core/General/BCBuilder.hpp"
#include "feddlib/problems/abstract/Problem_decl.hpp"
#include <FROSch_TpetraPreconditioner_decl.hpp>
#include <Teuchos_Assert.hpp>

/*!
 Definition of NonLinearSolver

 @brief  NonLinearSolver
 @author Christian Hochmuth
 @version 1.0
 @copyright CH
 */

namespace FEDD {


template<class SC,class LO,class GO,class NO>
NonLinearSolver<SC,LO,GO,NO>::NonLinearSolver():
type_("")
{}


template<class SC,class LO,class GO,class NO>
NonLinearSolver<SC,LO,GO,NO>::NonLinearSolver(std::string type):
type_(type)
{}

template<class SC,class LO,class GO,class NO>
NonLinearSolver<SC,LO,GO,NO>::~NonLinearSolver(){

}

template<class SC,class LO,class GO,class NO>
void NonLinearSolver<SC,LO,GO,NO>::solve(NonLinearProblem_Type &problem,vec_dbl_ptr_Type valuesForExport){

    if (!type_.compare("FixedPoint")) {
        solveFixedPoint(problem,valuesForExport);
    }
    else if(!type_.compare("Newton")){
        solveNewton(problem,valuesForExport);
    }
    else if(!type_.compare("FixedPointNewton")){
        solveFixedPointNewton(problem, valuesForExport);
    }
    else if(!type_.compare("NOX")){
#ifdef FEDD_HAVE_NOX
        solveNOX(problem,valuesForExport);
#endif
    } else if (!type_.compare("NonLinearSchwarz")) {
        solveNonLinearSchwarz(problem);
    }

}

template<class SC,class LO,class GO,class NO>
void NonLinearSolver<SC,LO,GO,NO>::solve(TimeProblem_Type &problem, double time, vec_dbl_ptr_Type valuesForExport){

    if (!type_.compare("FixedPoint")) {
        solveFixedPoint(problem,time);
    }
    else if(!type_.compare("Newton")){
        solveNewton(problem,time, valuesForExport);
    }
    else if(!type_.compare("NOX")){
#ifdef FEDD_HAVE_NOX
        solveNOX(problem, valuesForExport);
#endif
    }
    else if(!type_.compare("Extrapolation")){
        solveExtrapolation(problem, time);
    }
}

#ifdef FEDD_HAVE_NOX
template<class SC,class LO,class GO,class NO>
void NonLinearSolver<SC,LO,GO,NO>::solveNOX(NonLinearProblem_Type &problem,vec_dbl_ptr_Type valuesForExport){

    bool verbose = problem.getVerbose();
    Teuchos::RCP<NonLinearProblem<SC,LO,GO,NO> > problemPtr = Teuchos::rcpFromRef(problem);
    Teuchos::RCP<Teuchos::ParameterList> p = sublist(problemPtr->getParameterList(),"ThyraSolver");
//    p->set("Preconditioner Type", "None"); // CH 16.04.19: preconditioner will be built seperately
//    sublist( sublist(p, "Linear Solver Types") , "Belos")->set("Left Preconditioner If Unspecified",true);
    problemPtr->getLinearSolverBuilder()->setParameterList(p);

    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<SC> > lowsFactory = problemPtr->getLinearSolverBuilder()->createLinearSolveStrategy("");

	//problemPtr->set_W_factory(lowsFactory);

    // Create the initial guess
    Teuchos::RCP<Thyra::VectorBase<SC> > initial_guess = problemPtr->getNominalValues().get_x()->clone_v();
    Thyra::V_S(initial_guess.ptr(),Teuchos::ScalarTraits<SC>::zero());

    Teuchos::RCP<Thyra::LinearOpBase<SC> > W_op = problemPtr->create_W_op();
    Teuchos::RCP<Thyra::PreconditionerBase<SC> > W_prec = problemPtr->create_W_prec();

    // cout << " NonLinearSolver<SC,LO,GO,NO>::solveNOX " << endl;
    
    // problemPtr->create_W_op();

    Teuchos::RCP<NOX::Thyra::Group> nox_group(new NOX::Thyra::Group(initial_guess,
                                                                    problemPtr.getConst(),
                                                                    W_op,
                                                                    lowsFactory.getConst(),
                                                                    W_prec,
                                                                    Teuchos::null,
                                                                    Teuchos::null,
                                                                    Teuchos::null));

    nox_group->computeF();

    // Create the NOX status tests and the solver
    // Create the convergence tests

    Teuchos::RCP<NOX::StatusTest::NormUpdate> updateTol =
        Teuchos::rcp(new NOX::StatusTest::NormUpdate( problemPtr->getParameterList()->sublist("Parameter").get("updateTol",1.0e-6) ) );
    
    Teuchos::RCP<NOX::StatusTest::RelativeNormF> relresid =
        Teuchos::rcp(new NOX::StatusTest::RelativeNormF( problemPtr->getParameterList()->sublist("Parameter").get("relNonLinTol",1.0e-4) ) );
    
    Teuchos::RCP<NOX::StatusTest::NormWRMS> wrms =
        Teuchos::rcp(new NOX::StatusTest::NormWRMS(problemPtr->getParameterList()->sublist("Parameter").get("relNonLinTol",1.0e-4), problemPtr->getParameterList()->sublist("Parameter").get("absNonLinTol",1.0e-6)));
    
    Teuchos::RCP<NOX::StatusTest::NormF> absRes =
        Teuchos::rcp(new NOX::StatusTest::NormF( problemPtr->getParameterList()->sublist("Parameter").get("absNonLinTol",1.0e-6) ) );
    
    Teuchos::RCP<NOX::StatusTest::Combo> converged;
    
    if ( !problemPtr->getParameterList()->sublist("Parameter").get("Combo","AND").compare("AND") )
        converged = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
    else if (!problemPtr->getParameterList()->sublist("Parameter").get("Combo","AND").compare("OR") )
        converged = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
    
    if ( problemPtr->getParameterList()->sublist("Parameter").get("Use rel tol",true) )
        converged->addStatusTest(relresid);
    if ( problemPtr->getParameterList()->sublist("Parameter").get("Use update tol",false) )
        converged->addStatusTest(updateTol);
    if (problemPtr->getParameterList()->sublist("Parameter").get("Use WRMS",false))
        converged->addStatusTest(wrms);
    if (problemPtr->getParameterList()->sublist("Parameter").get("Use abs tol",false))
        converged->addStatusTest(absRes);

    Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters =
    Teuchos::rcp(new NOX::StatusTest::MaxIters(problemPtr->getParameterList()->sublist("Parameter").get("MaxNonLinIts",10)));
    Teuchos::RCP<NOX::StatusTest::FiniteValue> fv =
    Teuchos::rcp(new NOX::StatusTest::FiniteValue);
    Teuchos::RCP<NOX::StatusTest::Combo> combo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
    combo->addStatusTest(fv);
    combo->addStatusTest(converged);
    combo->addStatusTest(maxiters);

    // Create nox parameter list
    Teuchos::RCP<Teuchos::ParameterList> nl_params = sublist(problemPtr->getParameterList(),"NOXSolver");

    // Create the solver
    Teuchos::RCP<NOX::Solver::Generic> solver = NOX::Solver::buildSolver(nox_group, combo, nl_params);
    NOX::StatusTest::StatusType solveStatus = solver->solve();
    double nonLinearIts = solver->getSolverStatistics()->linearSolve.allNonlinearSolves_NumLinearSolves;
    double linearIts = solver->getSolverStatistics()->linearSolve.allNonlinearSolves_NumLinearIterations;

    linearIts/=nonLinearIts;
    if (verbose){
        std::cout << "############################################################" << std::endl;
        std::cout << "### Total nonlinear iterations : " << nonLinearIts << "  with an average of " << linearIts << " linear iterations ###" << std::endl;
        std::cout << "############################################################" << std::endl;
    }
    
    if ( problemPtr->getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts",false) ) {
        TEUCHOS_TEST_FOR_EXCEPTION((int)nonLinearIts == problemPtr->getParameterList()->sublist("Parameter").get("MaxNonLinIts",10) ,std::runtime_error,"Maximum nonlinear Iterations reached. Problem might have converged in the last step. Still we cancel here.");
    }
	nonLinearIts_ = nonLinearIts;

}
    
template<class SC,class LO,class GO,class NO>
void NonLinearSolver<SC,LO,GO,NO>::solveNOX(TimeProblem_Type &problem, vec_dbl_ptr_Type valuesForExport){
    
    bool verbose = problem.getVerbose();
    Teuchos::RCP<TimeProblem_Type> problemPtr = Teuchos::rcpFromRef(problem);
    Teuchos::RCP<Teuchos::ParameterList> p = sublist(problemPtr->getParameterList(),"ThyraSolver");

    problemPtr->getLinearSolverBuilder()->setParameterList(p);
    
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<SC> >
    lowsFactory = problemPtr->getLinearSolverBuilder()->createLinearSolveStrategy("");
    
    TEUCHOS_TEST_FOR_EXCEPTION(problemPtr->getSolution()->getNumVectors()>1, std::runtime_error, "With the current implementation NOX can only be used with 1 MultiVector column.");
    // Create the initial guess and fill with last solution
    Teuchos::RCP<Thyra::VectorBase<SC> > initialGuess = problemPtr->getNominalValues().get_x()->clone_v();
    // Try to convert to a ProductVB. If resulting pointer is not null we need to use the ProductMV below, otherwise it is a monolithic vector.
    Teuchos::RCP<Thyra::ProductVectorBase<SC> > initialGuessProd = Teuchos::rcp_dynamic_cast<Thyra::ProductVectorBase<SC> >(initialGuess);
    Teuchos::RCP<Thyra::MultiVectorBase<SC> > solMV;
    if (!initialGuessProd.is_null())
        solMV = problemPtr->getSolution()->getProdThyraMultiVector();
    else
        solMV = problemPtr->getSolution()->getThyraMultiVector();

    Thyra::assign(initialGuess.ptr(), *solMV->col(0));
    //Thyra::V_S(initialGuess.ptr(),Teuchos::ScalarTraits<SC>::zero());
    Teuchos::RCP<Thyra::LinearOpBase<SC> > W_op = problemPtr->create_W_op();
    Teuchos::RCP<Thyra::PreconditionerBase<SC> > W_prec = problemPtr->create_W_prec();
    // problemPtr->create_W_op();

    Teuchos::RCP<NOX::Thyra::Group> nox_group(new NOX::Thyra::Group(initialGuess,
                                                                    problemPtr.getConst(),
                                                                    W_op,
                                                                    lowsFactory.getConst(),
                                                                    W_prec,
                                                                    Teuchos::null,
                                                                    Teuchos::null,
                                                                    Teuchos::null));
    
    nox_group->computeF();
    
   // Create the NOX status tests and the solver
    // Create the convergence tests
    Teuchos::RCP<NOX::StatusTest::NormUpdate> updateTol =
        Teuchos::rcp(new NOX::StatusTest::NormUpdate( problemPtr->getParameterList()->sublist("Parameter").get("updateTol",1.0e-6) ) );
    
    Teuchos::RCP<NOX::StatusTest::RelativeNormF> relresid =
        Teuchos::rcp(new NOX::StatusTest::RelativeNormF( problemPtr->getParameterList()->sublist("Parameter").get("relNonLinTol",1.0e-4) ) );
    
    Teuchos::RCP<NOX::StatusTest::NormWRMS> wrms =
        Teuchos::rcp(new NOX::StatusTest::NormWRMS(problemPtr->getParameterList()->sublist("Parameter").get("relNonLinTol",1.0e-4), problemPtr->getParameterList()->sublist("Parameter").get("absNonLinTol",1.0e-6)));
    
    Teuchos::RCP<NOX::StatusTest::NormF> absRes =
        Teuchos::rcp(new NOX::StatusTest::NormF( problemPtr->getParameterList()->sublist("Parameter").get("absNonLinTol",1.0e-6) ) );
    
    Teuchos::RCP<NOX::StatusTest::Combo> converged;
    
    if ( !problemPtr->getParameterList()->sublist("Parameter").get("Combo","AND").compare("AND") )
        converged = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
    else if (!problemPtr->getParameterList()->sublist("Parameter").get("Combo","AND").compare("OR") )
        converged = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
    
    if ( problemPtr->getParameterList()->sublist("Parameter").get("Use rel tol",true) )
        converged->addStatusTest(relresid);
    if ( problemPtr->getParameterList()->sublist("Parameter").get("Use update tol",false) )
        converged->addStatusTest(updateTol);
    if (problemPtr->getParameterList()->sublist("Parameter").get("Use WRMS",false))
        converged->addStatusTest(wrms);
    if (problemPtr->getParameterList()->sublist("Parameter").get("Use abs tol",false))
        converged->addStatusTest(absRes);

    Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters =
    Teuchos::rcp(new NOX::StatusTest::MaxIters(problemPtr->getParameterList()->sublist("Parameter").get("MaxNonLinIts",10)));
    Teuchos::RCP<NOX::StatusTest::FiniteValue> fv =
    Teuchos::rcp(new NOX::StatusTest::FiniteValue);
    Teuchos::RCP<NOX::StatusTest::Combo> combo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
    combo->addStatusTest(fv);
    combo->addStatusTest(converged);
    combo->addStatusTest(maxiters);
    
    // Create nox parameter list
    Teuchos::RCP<Teuchos::ParameterList> nl_params = sublist(problemPtr->getParameterList(),"NOXSolver");
    // Create the solver
    Teuchos::RCP<NOX::Solver::Generic> solver =
    NOX::Solver::buildSolver(nox_group, combo, nl_params);
    NOX::StatusTest::StatusType solveStatus = solver->solve();
    
    double nonLinearIts = solver->getSolverStatistics()->linearSolve.allNonlinearSolves_NumLinearSolves;
    double linearIts = solver->getSolverStatistics()->linearSolve.allNonlinearSolves_NumLinearIterations;
    
    linearIts/=nonLinearIts;
    if (verbose){
        std::cout << "############################################################" << std::endl;
        std::cout << "### Total nonlinear iterations : " << nonLinearIts << "  with an average of " << linearIts << " linear iterations ###" << std::endl;
        std::cout << "############################################################" << std::endl;
    }
    
    if ( problemPtr->getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts",false) ) {
        TEUCHOS_TEST_FOR_EXCEPTION((int)nonLinearIts == problemPtr->getParameterList()->sublist("Parameter").get("MaxNonLinIts",10) ,std::runtime_error,"Maximum nonlinear Iterations reached. Problem might have converged in the last step. Still we cancel here.");
    }
    
    if (!valuesForExport.is_null()) {
        if (valuesForExport->size() == 2){
            (*valuesForExport)[0] = linearIts;
            (*valuesForExport)[1] = nonLinearIts;
        }
    }
}
#endif

template<class SC,class LO,class GO,class NO>
void NonLinearSolver<SC,LO,GO,NO>::solveFixedPoint(NonLinearProblem_Type &problem,vec_dbl_ptr_Type valuesForExport){

    bool verbose = problem.getVerbose();
    TEUCHOS_TEST_FOR_EXCEPTION(problem.getRhs()->getNumVectors()!=1,std::logic_error,"We need to change the code for numVectors>1.");
    // -------
    // fix point iteration
    // -------
    double	gmresIts = 0.;
    double residual0 = 1.;
    double residual = 1.;
    
    double tol = problem.getParameterList()->sublist("Parameter").get("relNonLinTol",1.0e-6);
    int maxNonLinIts = problem.getParameterList()->sublist("Parameter").get("MaxNonLinIts",10);
    int nlIts=0;

    double criterionValue = 1.;
    std::string criterion = problem.getParameterList()->sublist("Parameter").get("Criterion","Residual");


    while ( nlIts < maxNonLinIts ) {

        // This will compute the residual vector
        problem.calculateNonLinResidualVec("reverse");

        // Analogous to Newton but we assemble here only the parts which are needed for Fixed Point Iteration 
        problem.assemble("FixedPoint");

        problem.setBoundariesSystem();
        
        if (criterion=="Residual")
            residual = problem.calculateResidualNorm();
        
        if (nlIts==0)
            residual0 = residual;
    
        if (criterion=="Residual"){
            criterionValue = residual/residual0;
            if (verbose)
                std::cout << "### Fixed Point iteration : " << nlIts << "  relative nonlinear residual : " << criterionValue << std::endl;
            if ( criterionValue < tol )
                break;
        }

        // Set the current Nonlinear Step in the Problem class
        problem.setNonlinearIterationStep(nlIts);
        gmresIts += problem.solveAndUpdate( criterion, criterionValue );
        nlIts++;
        if(criterion=="Update"){
            if (verbose)
                std::cout << "### Fixed Point iteration : " << nlIts << "  residual of update : " << criterionValue << std::endl;
            if ( criterionValue < tol )
                break;
        }
        // ####### end FPI #######
    }

    gmresIts/=nlIts;
    if (verbose)
        std::cout << "### Total FPI : " << nlIts << "  with average gmres its : " << gmresIts << std::endl;
    if ( problem.getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts",false) ) {
        TEUCHOS_TEST_FOR_EXCEPTION( nlIts == maxNonLinIts ,std::runtime_error,"Maximum nonlinear Iterations reached. Problem might have converged in the last step. Still we cancel here.");
    }
    
}

template<class SC,class LO,class GO,class NO>
void NonLinearSolver<SC,LO,GO,NO>::solveNewton( NonLinearProblem_Type &problem, vec_dbl_ptr_Type valuesForExport ){

    bool verbose = problem.getVerbose();

    TEUCHOS_TEST_FOR_EXCEPTION(problem.getRhs()->getNumVectors()!=1,std::logic_error,"We need to change the code for numVectors>1.")
    // -------
    // Newton
    // -------
    double	gmresIts = 0.;
    double residual0 = 1.;
    double residual = 1.;
    double tol = problem.getParameterList()->sublist("Parameter").get("relNonLinTol",1.0e-6);
    int nlIts=0;
    int maxNonLinIts = problem.getParameterList()->sublist("Parameter").get("MaxNonLinIts",10);
    double criterionValue = 1.;
    std::string criterion = problem.getParameterList()->sublist("Parameter").get("Criterion","Residual");
    bool useBT = problem.getParameterList()->sublist("Parameter").get("Use Backtracking",true);

    while ( nlIts < maxNonLinIts ) {
        //this makes only sense for Navier-Stokes/Stokes, for other problems, e.g., non linear elasticity, it should do nothing.

        problem.calculateNonLinResidualVec("reverse");

        if (criterion=="Residual")
            residual = problem.calculateResidualNorm();

        problem.assemble("Newton");

        problem.setBoundariesSystem();

        if (nlIts==0)
            residual0 = residual;
        
        if (criterion=="Residual"){
            criterionValue = residual/residual0;
            if (verbose)
                std::cout << "### Newton iteration : " << nlIts << "  relative nonlinear residual : " << criterionValue << std::endl;
            if ( criterionValue < tol )
                break;
        }

        // Set the current Nonlinear Step in the Problem class
        problem.setNonlinearIterationStep(nlIts);
        gmresIts += problem.solveAndUpdate( criterion, criterionValue, problem.getComm()->getRank(), useBT);
        nlIts++;
        if(criterion=="Update"){
            if (verbose)
                std::cout << "### Newton iteration : " << nlIts << "  residual of update : " << criterionValue << std::endl;
            if ( criterionValue < tol )
                break;
        }

        // ####### end FPI #######
    }

    gmresIts/=nlIts;
    if (verbose)
        std::cout << "### Total Newton iterations : " << nlIts << "  with average gmres its : " << gmresIts << std::endl;
    if ( problem.getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts",false) ) {
        TEUCHOS_TEST_FOR_EXCEPTION(nlIts == maxNonLinIts ,std::runtime_error,"Maximum nonlinear Iterations reached. Problem might have converged in the last step. Still we cancel here.");
    }

    if (!valuesForExport.is_null()) {
        if (valuesForExport->size() == 2){
            (*valuesForExport)[0] = gmresIts;
            (*valuesForExport)[1] = nlIts;
        }
    }
}


/*
@TODO: In future user would like to switch between linearization based on the fact that the criteriumValue is not
       decreasing "enough" or that the it is increasing -> For this one would need to store the previous criterionValues in a vector
*/
template<class SC,class LO,class GO,class NO>
void NonLinearSolver<SC,LO,GO,NO>::solveFixedPointNewton( NonLinearProblem_Type &problem, vec_dbl_ptr_Type valuesForExport ){

    bool verbose = problem.getVerbose();

    TEUCHOS_TEST_FOR_EXCEPTION(problem.getRhs()->getNumVectors()!=1,std::logic_error,"We need to change the code for numVectors>1.")
    // -------
    // FixedPoint-Newton
    // -------
    double	gmresIts = 0.;
    double residual0 = 1.;
    double residual = 1.;
    double tol = problem.getParameterList()->sublist("Parameter").get("relNonLinTol",1.0e-6);
    int nlIts=0;
    int maxNonLinIts = problem.getParameterList()->sublist("Parameter").get("MaxNonLinIts",10);
    double criterionValue = 1.;
    std::string criterion = problem.getParameterList()->sublist("Parameter").get("Criterion","Residual");

    std::string linearization = problem.getParameterList()->sublist("General").get("Initial_Linearization", "FixedPoint"); // Default is FixedPoint for first iteration
    Teuchos::RCP<NonLinearProblem<SC,LO,GO,NO> > problemPtr = Teuchos::rcpFromRef(problem);
    Teuchos::RCP<Teuchos::ParameterList> parameterListGeneral = sublist(problemPtr->getParameterList(),"General");
    bool linearizationChanged = false;


    if (verbose)
        std::cout << "####### Run FixedPoint-Newton Combination with Initial Linearization: " << linearization << " ####### " << std::endl;

    while ( nlIts < maxNonLinIts ) { // User still can decinde on a maximum number of nonlinear iterations

        // Compute residual vector 
        problem.calculateNonLinResidualVec("reverse");    // Based on current solution (which is in first iteration the initial guess) compute the nonlinear residual vector

        if (criterion=="Residual")
            residual = problem.calculateResidualNorm();   // Compute norm of nonlinear residual vector


        if (nlIts==0) 
            residual0 = residual;                         // Store initial residual for relative convergence check
        
        if (criterion=="Residual"){                       // Check convergence based on relative nonlinear residual
            criterionValue = residual/residual0;
            if (verbose)
                std::cout << "### " << linearization << "  iteration : " << nlIts << "  relative nonlinear residual : " << criterionValue << std::endl;
            // One could add here a check for if nlIts=0 then initial linearization is shown which is not used for assemble?
            if ( criterionValue < tol )
                break;
        }

        // Use the user-defined switching strategy to decide whether to switch linearization and which one to use
        linearizationChanged = this->switchingStrategy_(linearization, nlIts, criterionValue, parameterListGeneral); // Get the next linearization type
        if (linearizationChanged) 
        {
            if (verbose)
                std::cout << "### Switching linearization to " << linearization << " in iteration " << nlIts << std::endl;
            problem.changeAssFELinearization(linearization); // Change linearization in all elements based on chosen strategy (Picard/ Newton) - But only if different from previous one
        }

        problem.assemble(linearization);


        problem.setBoundariesSystem();


        // PRINT INFOS
        // Set the current Nonlinear Step in the Problem class
        problem.setNonlinearIterationStep(nlIts);
        gmresIts += problem.solveAndUpdate( criterion, criterionValue );
        nlIts++;
        if(criterion=="Update"){
            if (verbose)
                std::cout << "### " << linearization << " iteration : " << nlIts << "  residual of update : " << criterionValue << std::endl;
            if ( criterionValue < tol )
                break;
        }

        // ####### end FPI #######
    }

    gmresIts/=nlIts;
    if (verbose)
        std::cout << "### Total " << "nonlinear" << "  iterations : " << nlIts << "  with average gmres its : " << gmresIts << std::endl;
    if ( problem.getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts",false) ) {
        TEUCHOS_TEST_FOR_EXCEPTION(nlIts == maxNonLinIts ,std::runtime_error,"Maximum nonlinear Iterations reached. Problem might have converged in the last step. Still we cancel here.");
    }

    if (!valuesForExport.is_null()) {
        if (valuesForExport->size() == 2){
            (*valuesForExport)[0] = gmresIts;
            (*valuesForExport)[1] = nlIts;
        }
    }
}




template<class SC,class LO,class GO,class NO>
void NonLinearSolver<SC,LO,GO,NO>::solveFixedPoint(TimeProblem_Type &problem, double time){

    bool verbose = problem.getVerbose();
    problem.setBoundariesRHS(time);
    TEUCHOS_TEST_FOR_EXCEPTION(problem.getRhs()->getNumVectors()!=1,std::logic_error,"We need to change the code for numVectors>1.")

    // -------
    // fix point iteration
    // -------
    double	gmresIts = 0.;
    double residual0 = 1.;
    double residual = 1.;
    double tol = problem.getParameterList()->sublist("Parameter").get("relNonLinTol",1.0e-6);
    int nlIts=0;
    int maxNonLinIts = problem.getParameterList()->sublist("Parameter").get("MaxNonLinIts",10);
    double criterionValue = 1.;
    std::string criterion = problem.getParameterList()->sublist("Parameter").get("Criterion","Residual");

    while ( nlIts < maxNonLinIts ) {
        
        problem.calculateNonLinResidualVec("reverse", time);

        if (criterion=="Residual")
            residual = problem.calculateResidualNorm();

        if (nlIts==0)
            residual0 = residual;
                    
        // Linearization of system matrix is done in calculateNonLinResidualVec
        // Now we need to combine it with the mass matrix
        problem.combineSystems();
        
        problem.setBoundariesSystem();
        
        if (criterion=="Residual"){
            criterionValue = residual/residual0;
            if (verbose)
                std::cout << "### Fixed Point iteration : " << nlIts << "  relative nonlinear residual : " << criterionValue << std::endl;
            if ( criterionValue < tol )
                break;
        }

        gmresIts += problem.solveAndUpdate( criterion, criterionValue );
        
        nlIts++;
        if(criterion=="Update"){
            if (verbose)
                std::cout << "### Fixed Point iteration : " << nlIts << "  residual of update : " << criterionValue << std::endl;
            if ( criterionValue < tol )
                break;
        }
        // ####### end FPI #######
    }
    
    gmresIts/=nlIts;
    if (verbose)
        std::cout << "### Total FPI : " << nlIts << "  with average gmres its : " << gmresIts << std::endl;
    if ( problem.getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts",false) ) {
        TEUCHOS_TEST_FOR_EXCEPTION( nlIts == maxNonLinIts ,std::runtime_error,"Maximum nonlinear Iterations reached. Problem might have converged in the last step. Still we cancel here.");
    }
}



template<class SC,class LO,class GO,class NO>
void NonLinearSolver<SC,LO,GO,NO>::solveNewton(TimeProblem_Type &problem, double time, vec_dbl_ptr_Type valuesForExport ){

    bool verbose = problem.getVerbose();
    problem.setBoundariesRHS(time);


    TEUCHOS_TEST_FOR_EXCEPTION(problem.getRhs()->getNumVectors()!=1,std::logic_error,"We need to change the code for numVectors>1.")
    
    // -------
    // Newton iteration
    // -------
    double	gmresIts = 0.;
    double residual0 = 1.;
    double residual = 1.;
    double tol = problem.getParameterList()->sublist("Parameter").get("relNonLinTol",1.0e-6);
    int nlIts=0;
    int maxNonLinIts = problem.getParameterList()->sublist("Parameter").get("MaxNonLinIts",10);
    double criterionValue = 1.;
    std::string criterion = problem.getParameterList()->sublist("Parameter").get("Criterion","Residual");
    std::string timestepping = problem.getParameterList()->sublist("Timestepping Parameter").get("Class","Singlestep");

    while ( nlIts < maxNonLinIts ) {
        if (timestepping == "External")
            problem.calculateNonLinResidualVec("external", time);
        else
            problem.calculateNonLinResidualVec("reverse", time);
        if (criterion=="Residual")
            residual = problem.calculateResidualNorm();
        
        if (nlIts==0)
            residual0 = residual;
        
        if (criterion=="Residual"){
            criterionValue = residual/residual0;
//            exporterTxt->exportData( criterionValue );
            if (verbose)
                std::cout << "### Newton iteration : " << nlIts << "  relative nonlinear residual : " << criterionValue << std::endl;
            if ( criterionValue < tol )
                break;
        }

        // Systems are combined in timeProblem.assemble("Newton") and then combined
        problem.assemble("Newton"); 

        problem.setBoundariesSystem();


        if (timestepping == "External"){//AceGen
            gmresIts += problem.solveAndUpdate( "ResidualAceGen", criterionValue );
        //    exporterTxt->exportData( criterionValue );

            //problem.assembleExternal( "OnlyUpdate" );// update AceGEN internal variables
        }
        else
            gmresIts += problem.solveAndUpdate( criterion, criterionValue );
        
        nlIts++;

        //problem.getSolution()->getBlock(0)->print();
        if(criterion=="Update"){
            if (verbose)
                std::cout << "### Newton iteration : " << nlIts << "  residual of update : " << criterionValue << std::endl;
            if ( criterionValue < tol )
                break;
        }

        // ####### end FPI #######
    }

    gmresIts/=nlIts;
    if (verbose)
        std::cout << "### Total Newton iteration : " << nlIts << "  with average gmres its : " << gmresIts << std::endl;
    if ( problem.getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts",false) ) {
        TEUCHOS_TEST_FOR_EXCEPTION(nlIts == maxNonLinIts ,std::runtime_error,"Maximum nonlinear Iterations reached. Problem might have converged in the last step. Still we cancel here.");
    }
    if (!valuesForExport.is_null()) {
        if (valuesForExport->size() == 2){
            (*valuesForExport)[0] = gmresIts;
            (*valuesForExport)[1] = nlIts;
        }
       

    }
    
}


template<class SC,class LO,class GO,class NO>
void NonLinearSolver<SC,LO,GO,NO>::solveExtrapolation(TimeProblem<SC,LO,GO,NO> &problem, double time){

    bool verbose = problem.getVerbose();

    problem.assemble("Extrapolation");

    problem.setBoundaries(time); // Setting boundaries to system rhs. The rest of the rhs (e.g. M*u_t) must/should be implemented in DAESolver

    int	gmresIts = problem.solve( );

    if (verbose) {
        std::cout << "### GMRES Its : " << gmresIts << std::endl;
    }
}

// Notes on using the nonlinear Schwarz solver:
//
// - Overlapping subdomain boundary nodes are marked with the boundary condition flag -99 during mesh partitioning. This
// information is required internally by the nonlinear Schwarz solver to set a dirichlet boundary condition
// during local subdomain solves. To facilitate this the following line must be included in the main.cpp
// bcFactory->addBC(currentSolutionDirichlet, -99, 0, domain, "Dirichlet", 1);
// together with the function
// void currentSolutionDirichlet(double *x, double *res, double t, const double *parameters) { res[0] = x[0]; }
//
// - The coarse space basis functions are built using the Jacobian evaluated at the initial solution. By default this is
// the zero vector. An initial solution of choice can be set by calling:
// problem->initSolutionWithFunction(initialValue2D, 0, std::vector<double>{0});
// together with e.g.
// void initialValue2D(double *x, double *res, double *parameters) { res[0] = x[0] * x[1] * (1 - x[0]) * (1 - x[1]); }
//
// - TODO: [KH] 25.07.25 Epetra and hence Xpetra are being deprecated in 2025. We have already migrated the FEDDLib to
// Tpetra. However, since the nonlinear Schwarz operators inherit from FROSch, which currently still uses Xpetra, we
// cannot migrate the nonlinear Schwarz implementation yet. For now, Tpetra and Xpetra conversions are done in
// solveNonLinearSchwarz() since this is where the nonlinear Schwarz operators and the FEDDLib "meet".
template <class SC, class LO, class GO, class NO>
void NonLinearSolver<SC, LO, GO, NO>::solveNonLinearSchwarz(NonLinearProblem_Type &problem) {

    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    print("###############################################################\n", problem.getComm());
    print("############ Starting nonlinear Schwarz solve ... #############\n", problem.getComm());
    print("###############################################################\n", problem.getComm());

    // If local pressure projection is to be used, modify the parameter list as required
    auto pressureProjection = problem.getPreconditioner()->getPressureProjection();
    if (problem.getParameterList()->sublist("Parameter").get("Use Pressure Projection", false)) {
        TEUCHOS_ASSERT(!pressureProjection.is_null())
        pressureProjection->merge(); // We merge the projection vector, as FROSch does not distinguish between blocks
        Teuchos::RCP< Tpetra::MultiVector<SC,LO,GO,NO> > vectorTpetra =  pressureProjection->getMergedVectorNonConst()->getTpetraMultiVectorNonConst();
        Teuchos::RCP< Xpetra::TpetraMultiVector<SC,LO,GO,NO> > vectorXpetraTpetra = Teuchos::rcp(new Xpetra::TpetraMultiVector<SC,LO,GO,NO>(vectorTpetra));
        Teuchos::RCP< Xpetra::MultiVector<SC,LO,GO,NO> > vectorXpetra = Teuchos::rcp_dynamic_cast<Xpetra::MultiVector<SC,LO,GO,NO>>(vectorXpetraTpetra);
        problem.getParameterList()->set("Projection", vectorXpetra);
    }
    // Define nonlinear Schwarz operator
    auto domainVec = problem.getDomainVector();
    auto mpiComm = domainVec.at(0)->getComm();
    auto serialComm = Teuchos::createSerialComm<LO>();

    auto useASPEN = problem.getParameterList()->get("Use ASPEN", true);

    auto mapOverlapping = Teuchos::rcp(new BlockMap_Type(0));
    auto mapOverlappingGhosts = Teuchos::rcp(new BlockMap_Type(0));
    auto mapUnique = Teuchos::rcp(new BlockMap_Type(0));
    int approxEntriesPerRow = 0;
    for (auto i = 0; i < domainVec.size(); i++) {
        approxEntriesPerRow = std::max(approxEntriesPerRow, domainVec.at(i)->getApproxEntriesPerRow());
        if (problem.getDofsPerNode(i) > 1) {
            mapOverlapping->addBlock(domainVec.at(i)->getMapVecFieldOverlapping(), i);
            mapOverlappingGhosts->addBlock(domainVec.at(i)->getMapVecFieldOverlappingGhosts(), i);
            mapUnique->addBlock(domainVec.at(i)->getMapVecFieldUnique(), i);
        } else {
            mapOverlapping->addBlock(domainVec.at(i)->getMesh()->getMapOverlapping(), i);
            mapOverlappingGhosts->addBlock(domainVec.at(i)->getMesh()->getMapOverlappingGhosts(), i);
            mapUnique->addBlock(domainVec.at(i)->getMesh()->getMapUnique(), i);
        }
    }
    auto mapOverlappingMerged = mapOverlapping->getMergedMap();
    auto mapOverlappingGhostsMerged = mapOverlappingGhosts->getMergedMap();
    auto mapUniqueMerged = mapUnique->getMergedMap();

    // The coarse space is built using this Jacobian
    // Calling these first two lines does have an affect on the Jacobian that is used to build the coarse space.
    // Depending on the particular problem implementation a different set of these calls to assemble should be used.
    problem.solution_->scale(-1.);
    problem.bcFactory_->setBCMinusVector(problem.solution_, problem.solution_);
    problem.assemble();
    problem.assemble("FixedPoint");
    problem.assemble("Newton");
    problem.setBoundariesSystem();

    // The operators
    auto nonLinearSchwarzOp = Teuchos::rcp(new FROSch::NonLinearSchwarzOperator<SC, LO, GO, NO>(
        serialComm, Teuchos::rcpFromRef(problem), problem.getParameterList()));
    nonLinearSchwarzOp->initialize();

    // When using auto, this is an rcp to a const NonLinearSumOperator. We need non-const here to ensure that the
    // non-const (nonlinear) apply() overload is called
    Teuchos::RCP<FROSch::NonLinearCombineOperator<SC, LO, GO, NO>> rhsCombineOperator;
    Teuchos::RCP<FROSch::CombineOperator<SC, LO, GO, NO>> simpleCombineOperator;
    auto variantString = problem.getParameterList()->get("Nonlin Schwarz Variant", "Additive");
    if (variantString == "Additive") {
        rhsCombineOperator = Teuchos::rcp(new FROSch::NonLinearSumOperator<SC, LO, GO, NO>(mpiComm));
        simpleCombineOperator = Teuchos::rcp(new FROSch::NewSumOperator<SC, LO, GO, NO>(mpiComm));
    } else if (variantString == "H1") {
        rhsCombineOperator = Teuchos::rcp(new FROSch::NonLinearH1Operator<SC, LO, GO, NO>(mpiComm));
        simpleCombineOperator = Teuchos::rcp(new FROSch::H1Operator<SC, LO, GO, NO>(mpiComm));
    } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Invalid nonlinear Schwarz variant");
    }

    rhsCombineOperator->addOperator(
        Teuchos::rcp_implicit_cast<FROSch::NonLinearOperator<SC, LO, GO, NO>>(nonLinearSchwarzOp));

    int numLevels = problem.getParameterList()->get("Levels", 1);
    TEUCHOS_TEST_FOR_EXCEPTION(numLevels > 2, std::runtime_error, "More than two levels are not implemented");
    auto coarseOperator = Teuchos::rcp(new FROSch::CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>(
        Teuchos::rcpFromRef(problem), sublist(problem.getParameterList(), "Coarse Nonlinear Schwarz")));
    if (numLevels == 2) {
        // This ensures that the coarse operator gets the required connectivity information for building the coarse space
        problem.assembleCoarseConnectivity();
        auto tempMat = Teuchos::rcp_const_cast<const Xpetra::Matrix<SC, LO, GO, NO>>(toXpetraMatrix(problem.system_->getMergedMatrix()->getTpetraMatrixNonConst()));
        coarseOperator->resetMatrix(tempMat);
        coarseOperator->initialize();
        // Remove the coarse connectivity again once we are done
        problem.removeCoarseConnectivity();
        problem.setBoundariesSystem();
        tempMat = Teuchos::rcp_const_cast<const Xpetra::Matrix<SC, LO, GO, NO>>(toXpetraMatrix(problem.system_->getMergedMatrix()->getTpetraMatrixNonConst()));
        coarseOperator->resetMatrix(tempMat);
        coarseOperator->compute();

        if (problem.getParameterList()->sublist("Exporter").get("Export coarse functions", false)) {
            coarseOperator->exportCoarseBasis();
        }
        rhsCombineOperator->addOperator(
            Teuchos::rcp_implicit_cast<FROSch::NonLinearOperator<SC, LO, GO, NO>>(coarseOperator));
    }

    // NOTE: [KH] 28.07.25 all the nonlinear Schwarz ops will continue working with Xpetra until FROSch migrates to Tpetra.
    auto simpleOverlappingOperator = Teuchos::rcp(new FROSch::SimpleOverlappingOperator<SC, LO, GO, NO>(
        Teuchos::rcpFromRef(problem), problem.getParameterList()));
    simpleCombineOperator->addOperator(simpleOverlappingOperator);

    auto simpleCoarseOperator = Teuchos::rcp(new FROSch::SimpleCoarseOperator<SC, LO, GO, NO>(
        toXpetraMatrix(problem.system_->getMergedMatrix()->getTpetraMatrixNonConst()), problem.getParameterList()));
    if (numLevels == 2) {
        simpleCoarseOperator->initialize(coarseOperator);
        simpleCombineOperator->addOperator(simpleCoarseOperator);
    }
    // Init (block) multi-vectors for nonlinear residual and outer Newton update
    auto g = Teuchos::rcp(new MultiVector_Type(mapUniqueMerged, 1));
    g->putScalar(0.);
    // deltaSolution needs to be a block multivector so it can update problem->solution_
    auto deltaSolution = Teuchos::rcp(new BlockMultiVector_Type(domainVec.size()));
    for (int i = 0; i < domainVec.size(); i++) {
        auto deltaSolutionBlock = Teuchos::rcp(new MultiVector_Type(mapUnique->getBlock(i), 1));
        deltaSolutionBlock->putScalar(0.);
        deltaSolution->addBlock(deltaSolutionBlock, i);
    }
    Teuchos::RCP<const Xpetra::Matrix<SC, LO, GO, NO>> localJacobian;
    Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> jacobianGhosts;
    Teuchos::RCP<Xpetra::Import<LO, GO>> uniqueToOverlappingGhostsImporter;
    ParameterListPtr_Type params = Teuchos::parameterList();
    if (!useASPEN) {
        jacobianGhosts = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(Xpetra::toXpetra(mapOverlappingGhostsMerged()->getTpetraMap()),
                                                                      approxEntriesPerRow);
        uniqueToOverlappingGhostsImporter = Xpetra::ImportFactory<LO, GO>::Build(
            Xpetra::toXpetra(mapUniqueMerged()->getTpetraMap()), Xpetra::toXpetra(mapOverlappingGhostsMerged()->getTpetraMap()));
        // So we can call doImport after calling resumeFill on a fillComplete matrix
        params->set("Optimize Storage", false);
    }

    // Define convergence requirements
    double gmresIts = 0.;
    double outerRelTol = problem.getParameterList()->sublist("Parameter").get("relNonLinTol", 1.0e-6);
    double outerAbsTol = problem.getParameterList()->sublist("Parameter").get("absNonLinTol", 1.0e-6);
    int outerNonLinIts = 0;
    int maxOuterNonLinIts = problem.getParameterList()->sublist("Parameter").get("MaxNonLinIts", 10);

    // Print solver settings
    // Print solver settings
    print("+++++++++ Nonlinear Schwarz solver configuration ++++++++++\n", mpiComm);
    print("==> Use ASPEN: " + std::to_string(useASPEN) + "\n", mpiComm);
    print("==> Variant: " + variantString + "\n", mpiComm);
    print("==> Combine mode: " + problem.getParameterList()->get("Combine Mode", "Restricted") + "\n", mpiComm);
    print("==> Overlap: " + std::to_string(problem.getParameterList()->get("Overlap", 1)) + "\n", mpiComm);
    print("==> Use Backtracking: " + std::to_string(problem.getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Use Backtracking",false)) + "\n", mpiComm);
    print("==> Remove Dirichlet Nodes: " + std::to_string(problem.getParameterList()->sublist("Coarse Nonlinear Schwarz").get("Remove Dirichlet Nodes",false)) + "\n", mpiComm);
    print("==> IPOU type block 1: " + problem.getParameterList()->sublist("Coarse Nonlinear Schwarz").sublist("Blocks").sublist("1").sublist("InterfacePartitionOfUnity").get("Type","Use must provide a IPOU type for block 1") + "\n", mpiComm);
    print("==> Distance function block 1: " + problem.getParameterList()->sublist("Coarse Nonlinear Schwarz").sublist("Blocks").sublist("1").sublist("InterfacePartitionOfUnity").sublist("RGDSW").get("Distance Function","None provided") + "\n", mpiComm);
    print("==> IPOU type block 2: " + problem.getParameterList()->sublist("Coarse Nonlinear Schwarz").sublist("Blocks").sublist("2").sublist("InterfacePartitionOfUnity").get("Type","Use must provide a IPOU type for block 2") + "\n", mpiComm);
    print("==> Distance function block 2: " + problem.getParameterList()->sublist("Coarse Nonlinear Schwarz").sublist("Blocks").sublist("2").sublist("InterfacePartitionOfUnity").sublist("RGDSW").get("Distance Function","None provided") + "\n", mpiComm);
    print("==> Rel. tol: " + std::to_string(outerRelTol) + "\n", mpiComm);
    print("==> Abs. tol: " + std::to_string(outerAbsTol) + "\n", mpiComm);
    print("==> Max outer Newton iters.: " + std::to_string(maxOuterNonLinIts) + "\n", mpiComm);
    print("++++++++++++++++++++++++++++++++++++++++++\n", mpiComm);

    // Compute the residual
    problem.calculateNonLinResidualVec("reverse");
    auto residual0 = problem.calculateResidualNorm();
    auto residual = residual0;
    auto relativeResidual = residual / residual0;

    // Outer Newton iterations
    while (relativeResidual > outerRelTol && residual > outerAbsTol && outerNonLinIts < maxOuterNonLinIts) {
        logGreen("Starting outer Newton iteration: " + std::to_string(outerNonLinIts), mpiComm);
        print("Rel. residual: ", mpiComm);
        print(relativeResidual, mpiComm);
        print(" Abs. residual: ", mpiComm);
        print(residual, mpiComm);
        print("\n", mpiComm);

        // Compute the residual of the alternative problem \mathcal{F} = g
        // g fulfills the boundary conditions
        logGreen("Computing nonlinear Schwarz operator", mpiComm);
        rhsCombineOperator->apply(*problem.getSolution()->getMergedVectorNonConst()->getTpetraMultiVectorNonConst(),
                                  *g->getTpetraMultiVectorNonConst());
        if (numLevels == 2) {
            // Update SimpleCoarseOperator to ensure it wraps the current CoarseNonLinearSchwarzOperator
            simpleCoarseOperator->initialize(coarseOperator);
        }
        if (useASPEN) {
            logGreen("Building ASPEN tangent", mpiComm);
            localJacobian = toXpetraMatrix(nonLinearSchwarzOp->getLocalJacobianGhosts()->getMergedMatrix()->getTpetraMatrixNonConst());
        } else {
            logGreen("Building ASPIN tangent", mpiComm);
            problem.assemble("Newton");
            problem.setBoundariesSystem();
            // Compute D\mathcal{F}(u) using FROSch and DF(u)
            auto jacobian = toXpetraMatrix(problem.getSystem()->getMergedMatrix()->getTpetraMatrixNonConst());
            jacobianGhosts->setAllToScalar(ST::zero());
            jacobianGhosts->resumeFill();
            jacobianGhosts->doImport(*jacobian, *uniqueToOverlappingGhostsImporter, Xpetra::ADD);
            jacobianGhosts->fillComplete(params);
            localJacobian = FROSch::ExtractLocalSubdomainMatrix(jacobianGhosts.getConst(),
                                                                Xpetra::toXpetra(mapOverlappingGhostsMerged()->getTpetraMap()));
        }
        simpleOverlappingOperator->initialize(serialComm, localJacobian, Xpetra::toXpetra(mapOverlappingMerged()->getTpetraMap()),
                                              Xpetra::toXpetra(mapOverlappingGhostsMerged()->getTpetraMap()),
                                              Xpetra::toXpetra(mapUniqueMerged()->getTpetraMap()));
        simpleOverlappingOperator->compute();
        // Convert SchwarzOperator to Thyra::LinearOpBase
        auto xpetraOverlappingOperator =
            Teuchos::rcp_static_cast<Xpetra::Operator<SC, LO, GO, NO>>(simpleCombineOperator);

        Teuchos::RCP<FROSch::TpetraPreconditioner<SC, LO, GO, NO>> tpetraFROSchOverlappingOperator(
            new FROSch::TpetraPreconditioner<SC, LO, GO, NO>(xpetraOverlappingOperator));

        auto tpetraOverlappingOperator =
            Teuchos::rcp_static_cast<Tpetra::Operator<SC, LO, GO, NO>>(tpetraFROSchOverlappingOperator);

        auto thyraOverlappingOperator = Thyra::createLinearOp(tpetraOverlappingOperator);

        // Solve linear system with GMRES
        logGreen("Solving outer linear system", mpiComm);
        FEDD_TIMER_START(GMRESTimer, " - Schwarz - GMRES solve");
        gmresIts += solveThyraLinOp(thyraOverlappingOperator, deltaSolution->getThyraMultiVector(),
                                    g->getThyraMultiVector(), problem.getParameterList());
        FEDD_TIMER_STOP(GMRESTimer);

        deltaSolution->split();
        // Update the current solution
        // solution = alpha * deltaSolution + beta * solution
        problem.solution_->update(-ST::one(), deltaSolution, ST::one());
        // Compute the residual
        problem.calculateNonLinResidualVec("reverse");
        residual = problem.calculateResidualNorm();
        relativeResidual = residual / residual0;

        outerNonLinIts += 1;
    }
    auto itersVecSubdomains = nonLinearSchwarzOp->getRunStats();
    auto itersVecCoarse = coarseOperator->getRunStats();
    print("================= Nonlinear Schwarz terminated =========================", mpiComm);
    print("\n\nOuter Newton:", mpiComm, 25);
    print("Rel. residual:", mpiComm, 15);
    print("Abs. residual:", mpiComm, 15);
    print("Iters:", mpiComm, 15);
    print("GMRES iters:", mpiComm, 15);
    print("\n", mpiComm, 25);
    print(relativeResidual, mpiComm, 15, 2);
    print(residual, mpiComm, 15, 2);
    print(outerNonLinIts, mpiComm, 15, 2);
    print(gmresIts, mpiComm, 15, 2);
    print("\n\nInner Newton:", mpiComm, 25);
    print("min. iters.", mpiComm, 15);
    print("mean iters.", mpiComm, 15);
    print("max. iters.", mpiComm, 15);
    print("\n", mpiComm, 25);
    print(itersVecSubdomains.at(0), mpiComm, 15, 2);
    print(itersVecSubdomains.at(1), mpiComm, 15, 2);
    print(itersVecSubdomains.at(2), mpiComm, 15, 2);
    if (numLevels == 2) {
        print("\n\nCoarse Newton:", mpiComm, 25);
        print("Iters.", mpiComm, 15);
        print("\n", mpiComm, 25);
        print(itersVecCoarse.at(0), mpiComm, 15, 2);
    }
    print("\n", mpiComm);
}

template <class SC, class LO, class GO, class NO>
int NonLinearSolver<SC, LO, GO, NO>::solveThyraLinOp(Teuchos::RCP<const Thyra::LinearOpBase<SC>> thyraLinOp,
                                                     Teuchos::RCP<Thyra::MultiVectorBase<SC>> thyraX, Teuchos::RCP<const Thyra::MultiVectorBase<SC>> thyraB,
                                                     ParameterListPtr_Type parameterList, bool verbose) {
    int its = 0;
    thyraX->assign(0.);

    auto pListThyraSolver =
        sublist(sublist(parameterList, "Outer Newton Nonlinear Schwarz"), "Thyra Solver Outer Newton");

    auto linearSolverBuilder = Teuchos::rcp(new Stratimikos::DefaultLinearSolverBuilder());
    linearSolverBuilder->setParameterList(pListThyraSolver);
    auto lowsFactory = linearSolverBuilder->createLinearSolveStrategy("");

    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    lowsFactory->setOStream(out);
    lowsFactory->setVerbLevel(Teuchos::VERB_LOW);

    auto solver = lowsFactory->createOp();
    Thyra::initializeOp<SC>(*lowsFactory, thyraLinOp, solver.ptr());

    Thyra::SolveStatus<SC> status = Thyra::solve<SC>(*solver, Thyra::NOTRANS, *thyraB, thyraX.ptr());

    if (verbose) {
        std::cout << status << std::endl;
    }
    if (!pListThyraSolver->get("Linear Solver Type", "Belos").compare("Belos")) {
        its = status.extraParameters->get("Belos/Iteration Count", 0);
    } else {
        its = 0;
    }
    return its;
}
} // namespace FEDD
#endif
