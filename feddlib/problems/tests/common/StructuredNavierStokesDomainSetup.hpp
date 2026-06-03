#ifndef FEDDLIB_PROBLEMS_TESTS_COMMON_STRUCTURED_NAVIER_STOKES_DOMAINS_SETUP_HPP
#define FEDDLIB_PROBLEMS_TESTS_COMMON_STRUCTURED_NAVIER_STOKES_DOMAINS_SETUP_HPP

#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/problems/tests/common/VerboseStructuredDomainStats.hpp"

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TestForException.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace FEDD {
namespace Problems {
namespace Tests {
namespace Common {

/** BFS only: number of MPI blocks along the step (see examples/steadyNavierStokes). Channel/LDC use 1. */
inline int structuredNavierStokesMinSubdomains(const std::string &meshType, double length) {
    if (!meshType.compare("structured_bfs")) {
        return static_cast<int>(2 * length + 1);
    }
    return 1;
}

/** Output of setupStructuredNavierStokesDomain (steadyNavierStokesNKS / steadyNavierStokesNLS). */
template <class SC, class LO, class GO, class NO>
struct StructuredNavierStokesDomainResult {
    int n{};
    Teuchos::RCP<Domain<SC, LO, GO, NO>> domainPressure;
    Teuchos::RCP<Domain<SC, LO, GO, NO>> domainVelocity;
};

/**
 * Build structured velocity/pressure domains for channel, lid-driven cavity (LDC), or backward-facing step (BFS).
 *
 * Exactly one of meshType = structured | structured_ldc | structured_bfs is supported per call.
 *
 * length / height:
 *   - Channel (structured): length = x extent, height = cross-section (2D y or 3D yz side).
 *   - LDC: same Domain ctor as channel. Height is passed for all three dimensions to get a cube.
 *   - BFS: length = step length; height is unused here (fixed BFS box geometry).
 *
 * Caller should print "-- Building Mesh ..." before calling when verbose.
 */
template <class SC, class LO, class GO, class NO>
inline StructuredNavierStokesDomainResult<SC, LO, GO, NO> setupStructuredNavierStokesDomain(
    const Teuchos::RCP<const Teuchos::Comm<int>> &comm, bool verbose, int dim, const std::string &meshType,
    int active_ranks, int mpi_ranks_coarse_solve, double length, double height, int m,
    const std::string &discPressure, const std::string &discVelocity) {
    StructuredNavierStokesDomainResult<SC, LO, GO, NO> out;

    const int min_number_subdomains = structuredNavierStokesMinSubdomains(meshType, length);
    const bool is_bfs = !meshType.compare("structured_bfs");
    const bool is_channel = !meshType.compare("structured");
    const bool is_ldc = !meshType.compare("structured_ldc");

    TEUCHOS_TEST_FOR_EXCEPTION(!is_bfs && !is_channel && !is_ldc, std::logic_error,
                               "Unsupported Mesh Type. Use 'structured', 'structured_ldc', or 'structured_bfs'.");

    TEUCHOS_TEST_FOR_EXCEPTION(active_ranks % min_number_subdomains != 0, std::logic_error,
                               "Wrong number of processors for structured mesh.");

    const double eps = 100 * Teuchos::ScalarTraits<double>::eps();
    int n = 0;
    int geometryFlag = -1;
    std::string geometryName;

    TEUCHOS_TEST_FOR_EXCEPTION(dim != 2 && dim != 3, std::logic_error,
                               "setupStructuredNavierStokesDomain: dim must be 2 or 3.");

    if (is_bfs) {
        // --- structured_bfs: pow-based n, fixed BFS box, buildMesh2D/3DBFS ---
        n = static_cast<int>(std::pow(active_ranks / static_cast<double>(min_number_subdomains), 1.0 / dim) + eps);

        if (dim == 2) {
            std::vector<double> x(2);
            x[0] = -1.0;
            x[1] = -1.0;
            out.domainPressure.reset(new Domain<SC, LO, GO, NO>(x, length + 1., 2., comm));
            out.domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, length + 1., 2., comm));
        } else {
            std::vector<double> x(3);
            x[0] = -1.0;
            x[1] = 0.0;
            x[2] = -1.0;
            out.domainPressure.reset(new Domain<SC, LO, GO, NO>(x, length + 1., 1., 2., comm));
            out.domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, length + 1., 1., 2., comm));
        }

        geometryFlag = 2;
        geometryName = "BFS";
    } else if (is_channel) {
        // --- structured (channel): ratio-based n, rectangular domain at origin ---
        TEUCHOS_TEST_FOR_EXCEPTION(height <= 0., std::logic_error, "Height must be > 0 for structured channel mesh.");
        TEUCHOS_TEST_FOR_EXCEPTION(length < height, std::logic_error,
                                   "For structured channel mesh, length must be >= height.");

        const double ratio = length / height;
        const double nFloating = (dim == 2) ? std::sqrt(active_ranks / ratio) : std::cbrt(active_ranks / ratio);
        const double nRounded = std::round(nFloating);
        const double tol = 1000.0 * Teuchos::ScalarTraits<double>::eps() * std::max(1.0, std::abs(nFloating));
        TEUCHOS_TEST_FOR_EXCEPTION(std::abs(nFloating - nRounded) > tol, std::logic_error,
                                   "Invalid rank/geometry combination for structured channel mesh."
                                   " Need activeRanks/(length/height) to be a perfect square/cube.");
        n = static_cast<int>(nRounded);
        TEUCHOS_TEST_FOR_EXCEPTION(n < 1, std::logic_error, "Computed invalid n for structured channel mesh.");

        const double expected_ranks = (dim == 2) ? ratio * n * n : ratio * n * n * n;
        TEUCHOS_TEST_FOR_EXCEPTION(std::abs(active_ranks - expected_ranks) >
                                       tol * std::max(1.0, std::abs(static_cast<double>(active_ranks))),
                                   std::logic_error,
                                   "Inconsistent activeRanks, length/height and n in structured channel mesh.");

        if (dim == 2) {
            std::vector<double> x(2);
            x[0] = 0.0;
            x[1] = 0.0;
            out.domainPressure.reset(new Domain<SC, LO, GO, NO>(x, length, height, comm));
            out.domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, length, height, comm));
        } else {
            std::vector<double> x(3);
            x[0] = 0.0;
            x[1] = 0.0;
            x[2] = 0.0;
            out.domainPressure.reset(new Domain<SC, LO, GO, NO>(x, length, height, height, comm));
            out.domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, length, height, height, comm));
        }

        geometryFlag = 1;
        // Domain registers this label as geoNumber 0 -> setGeometry2DRectangle + buildMesh2D (not length==height).
        geometryName = "Square";
    } else if (is_ldc) {
        // --- structured_ldc: uniform n on a full processor grid, unit-scale box at origin ---
        n = static_cast<int>(std::pow(active_ranks / static_cast<double>(min_number_subdomains), 1.0 / dim) + eps);

        if (dim == 2) {
            std::vector<double> x(2);
            x[0] = 0.0;
            x[1] = 0.0;
            out.domainPressure.reset(new Domain<SC, LO, GO, NO>(x, height, height, comm));
            out.domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, height, height, comm));
        } else {
            std::vector<double> x(3);
            x[0] = 0.0;
            x[1] = 0.0;
            x[2] = 0.0;
            out.domainPressure.reset(new Domain<SC, LO, GO, NO>(x, height, height, height, comm));
            out.domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, height, height, height, comm));
        }

        geometryFlag = 5;
        // Same Domain geometry name as channel; flagsOption 5 selects LDC BC flags.
        geometryName = "Square";
    } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Passed an invalid mesh type to setupStructuredNavierStokesDomain.");
    }

    TEUCHOS_TEST_FOR_EXCEPTION(n < 1, std::logic_error, "Computed invalid n for structured mesh.");

    if (is_bfs) {
        const double expected = min_number_subdomains * std::pow(static_cast<double>(n), dim);
        const double tol = 1000.0 * Teuchos::ScalarTraits<double>::eps() *
                           std::max(1.0, std::abs(static_cast<double>(active_ranks)));
        TEUCHOS_TEST_FOR_EXCEPTION(std::abs(active_ranks - expected) > tol, std::logic_error,
                                   "Inconsistent activeRanks, length and n in structured BFS mesh.");
    }

    if (verbose) {
        printVerboseStructuredDomainStats(dim, meshType, comm->getSize(), mpi_ranks_coarse_solve, active_ranks, length,
                                          height, n, m);
    }

    // flagsOption: 1 channel, 5 LDC, 2 BFS (surface/BC tagging). geometryName selects the Domain/MeshStructured builder.
    out.domainPressure->buildMesh(geometryFlag, geometryName, dim, discPressure, n, m, mpi_ranks_coarse_solve);
    out.domainVelocity->buildMesh(geometryFlag, geometryName, dim, discVelocity, n, m, mpi_ranks_coarse_solve);
    out.domainVelocity->preProcessMesh(true, false);

    out.n = n;
    return out;
}

} // namespace Common
} // namespace Tests
} // namespace Problems
} // namespace FEDD

#endif
