#ifndef FEDDLIB_PROBLEMS_TESTS_COMMON_VERBOSE_STRUCTURED_DOMAIN_STATS_HPP
#define FEDDLIB_PROBLEMS_TESTS_COMMON_VERBOSE_STRUCTURED_DOMAIN_STATS_HPP

#include <iostream>
#include <string>

namespace FEDD {
namespace Problems {
namespace Tests {
namespace Common {

/** Multi-line structured-mesh / domain summary for LDC and channel test drivers (rank 0 only). */
inline void printVerboseStructuredDomainStats(int dim, const std::string &meshType, int total_mpi_ranks,
                                              int mpi_ranks_coarse_solve, int active_ranks, double length,
                                              double height, int n, int m) {
    const double ratio = length / height;
    double expectedActiveRanks = 0.0;
    if (!meshType.compare("structured")) {
        expectedActiveRanks = (dim == 2) ? ratio * n * n : ratio * n * n * n;
    } else {
        expectedActiveRanks = (dim == 2) ? n * n : n * n * n;
    }

    std::cout << "\n-- Domain stats --" << std::endl;
    std::cout << "  dim                               : " << dim << " (domain dimension)" << std::endl;
    std::cout << "  meshType                          : " << meshType
              << " (structured=channel, structured_ldc=lid-driven cavity)" << std::endl;
    std::cout << "  total_mpi_ranks                   : " << total_mpi_ranks << std::endl;
    std::cout << "  mpi_ranks_coarse_solve            : " << mpi_ranks_coarse_solve
              << " (reserved for coarse solve; General / \"Mpi Ranks Coarse\")" << std::endl;
    std::cout << "  active_ranks                      : " << active_ranks
              << " (total_mpi_ranks - mpi_ranks_coarse_solve)" << std::endl;
    std::cout << "  length                            : " << length << " (domain extent in x)" << std::endl;
    std::cout << "  height                            : " << height
              << " (2D: y extent; 3D: yz-square side, width assumed equal)" << std::endl;
    std::cout << "  ratio (length / height)           : " << ratio
              << " (used with structured channel partitioning)" << std::endl;
    std::cout << "  n (MeshStructured subdomain count_yz) : " << n
              << " (input N: 2D => Ny=n, Nx from length; 3D => Ny=Nz=n, Nx from length)" << std::endl;
    std::cout << "  m (H/h)                           : " << m << " (elements per subdomain patch edge)"
              << std::endl;
    std::cout << "  expected_active_ranks_consistency : " << expectedActiveRanks
              << " (Nx*Ny in 2D channel or similar; should equal active_ranks)" << std::endl;
    std::cout << "-- end Domain stats --" << std::endl;
}

} // namespace Common
} // namespace Tests
} // namespace Problems
} // namespace FEDD

#endif
