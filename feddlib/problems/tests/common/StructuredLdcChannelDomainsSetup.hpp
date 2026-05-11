#ifndef FEDDLIB_PROBLEMS_TESTS_COMMON_STRUCTURED_LDC_CHANNEL_DOMAINS_SETUP_HPP
#define FEDDLIB_PROBLEMS_TESTS_COMMON_STRUCTURED_LDC_CHANNEL_DOMAINS_SETUP_HPP

#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/problems/tests/common/VerboseStructuredDomainStats.hpp"

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TestForException.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace FEDD {
namespace Problems {
namespace Tests {
namespace Common {

/** Output of setupStructuredLdcChannelDomains (steadyLDC / steadyLDCSchwarz structured paths). */
template <class SC, class LO, class GO, class NO>
struct StructuredLdcChannelDomainsResult {
    int n{};
    Teuchos::RCP<Domain<SC, LO, GO, NO>> domainPressure;
    Teuchos::RCP<Domain<SC, LO, GO, NO>> domainVelocity;
};

/**
 * Structured LDC or channel: compute subdomain count n, build velocity/pressure domains, print optional stats,
 * and call buildMesh / preProcessMesh. Caller should print "-- Building Mesh ..." when desired.
 */
template <class SC, class LO, class GO, class NO>
inline StructuredLdcChannelDomainsResult<SC, LO, GO, NO> setupStructuredLdcChannelDomains(
    const Teuchos::RCP<const Teuchos::Comm<int>> &comm, bool verbose, int dim, const std::string &meshType,
    int active_ranks, int mpi_ranks_coarse_solve, double length, double height, int m,
    const std::string &discPressure, const std::string &discVelocity, int min_number_subdomains) {
    StructuredLdcChannelDomainsResult<SC, LO, GO, NO> out;

    TEUCHOS_TEST_FOR_EXCEPTION(active_ranks % min_number_subdomains != 0, std::logic_error,
                               "Wrong number of processors for structured mesh.");

    int n = 0;
    if (dim == 2) {
        if (!meshType.compare("structured")) {
            TEUCHOS_TEST_FOR_EXCEPTION(height <= 0., std::logic_error, "Height must be > 0 for structured channel mesh.");
            TEUCHOS_TEST_FOR_EXCEPTION(length < height, std::logic_error,
                                       "For structured channel mesh in 2D, length must be >= height.");
            double ratio = length / height;
            double nFloating = std::sqrt(active_ranks / ratio);
            double nRounded = std::round(nFloating);
            double tol = 1000.0 * Teuchos::ScalarTraits<double>::eps() * std::max(1.0, std::abs(nFloating));
            TEUCHOS_TEST_FOR_EXCEPTION(std::abs(nFloating - nRounded) > tol, std::logic_error,
                                       "Invalid rank/geometry combination for 2D structured channel mesh."
                                       " Need activeRanks/(length/height) to be a perfect square.");
            n = static_cast<int>(nRounded);
            TEUCHOS_TEST_FOR_EXCEPTION(n < 1, std::logic_error, "Computed invalid n for structured channel mesh.");
            TEUCHOS_TEST_FOR_EXCEPTION(
                std::abs(active_ranks - ratio * n * n) > tol * std::max(1.0, std::abs(static_cast<double>(active_ranks))),
                std::logic_error, "Inconsistent activeRanks, length/height and n in 2D structured channel mesh.");
        } else {
            n = static_cast<int>(std::pow(active_ranks / min_number_subdomains, 1 / 2.) +
                                 100 * Teuchos::ScalarTraits<double>::eps());
        }
        std::vector<double> x(2);
        x[0] = 0.0;
        x[1] = 0.0;
        out.domainPressure.reset(new Domain<SC, LO, GO, NO>(x, length, height, comm));
        out.domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, length, height, comm));
    } else if (dim == 3) {
        if (!meshType.compare("structured")) {
            TEUCHOS_TEST_FOR_EXCEPTION(height <= 0., std::logic_error, "Height must be > 0 for structured channel mesh.");
            TEUCHOS_TEST_FOR_EXCEPTION(length < height, std::logic_error,
                                       "For structured channel mesh in 3D, length must be >= height.");
            double ratio = length / height;
            double nFloating = std::cbrt(active_ranks / ratio);
            double nRounded = std::round(nFloating);
            double tol = 1000.0 * Teuchos::ScalarTraits<double>::eps() * std::max(1.0, std::abs(nFloating));
            TEUCHOS_TEST_FOR_EXCEPTION(std::abs(nFloating - nRounded) > tol, std::logic_error,
                                       "Invalid rank/geometry combination for 3D structured channel mesh."
                                       " Need activeRanks/(length/height) to be a perfect cube.");
            n = static_cast<int>(nRounded);
            TEUCHOS_TEST_FOR_EXCEPTION(n < 1, std::logic_error, "Computed invalid n for structured channel mesh.");
            TEUCHOS_TEST_FOR_EXCEPTION(std::abs(active_ranks - ratio * n * n * n) >
                                           tol * std::max(1.0, std::abs(static_cast<double>(active_ranks))),
                                       std::logic_error,
                                       "Inconsistent activeRanks, length/height and n in 3D structured channel mesh.");
        } else {
            n = static_cast<int>(std::pow(active_ranks / min_number_subdomains, 1 / 3.) +
                                 100 * Teuchos::ScalarTraits<double>::eps());
        }
        std::vector<double> x(3);
        x[0] = 0.0;
        x[1] = 0.0;
        x[2] = 0.0;
        out.domainPressure.reset(new Domain<SC, LO, GO, NO>(x, length, height, height, comm));
        out.domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, length, height, height, comm));
    } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "setupStructuredLdcChannelDomains: dim must be 2 or 3.");
    }

    if (verbose) {
        printVerboseStructuredDomainStats(dim, meshType, comm->getSize(), mpi_ranks_coarse_solve, active_ranks, length,
                                          height, n, m);
    }

    int geometryFlag = -1;
    if (!meshType.compare("structured_ldc")) {
        geometryFlag = 5;
    } else if (!meshType.compare("structured")) {
        geometryFlag = 1;
    } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                   "Unsupported Mesh Type for this test. Use 'structured_ldc' or 'structured'.");
    }

    out.domainPressure->buildMesh(geometryFlag, "Square", dim, discPressure, n, m, mpi_ranks_coarse_solve);
    out.domainVelocity->buildMesh(geometryFlag, "Square", dim, discVelocity, n, m, mpi_ranks_coarse_solve);
    out.domainVelocity->preProcessMesh(true, false);

    out.n = n;
    return out;
}

} // namespace Common
} // namespace Tests
} // namespace Problems
} // namespace FEDD

#endif
