#ifndef FEDDLIB_PROBLEMS_TESTS_COMMON_VERBOSE_STRUCTURED_DOMAIN_STATS_HPP
#define FEDDLIB_PROBLEMS_TESTS_COMMON_VERBOSE_STRUCTURED_DOMAIN_STATS_HPP

#include <cmath>
#include <iostream>
#include <string>

namespace FEDD {
namespace Problems {
namespace Tests {
namespace Common {

/** Multi-line structured-mesh / domain summary for Navier-Stokes drivers (rank 0 only). */
inline void printVerboseStructuredDomainStats(int dim, const std::string &meshType, int total_mpi_ranks,
                                              int mpi_ranks_coarse_solve, int active_ranks, double length,
                                              double height, int n, int m) {
    const bool is_bfs = !meshType.compare("structured_bfs");
    const bool is_channel = !meshType.compare("structured");
    const int min_subdomains = is_bfs ? static_cast<int>(2 * length + 1) : 1;
    const int bfs_multiplier = is_bfs ? static_cast<int>(2 * (length + 1) - 1) : 0;

    double expectedActiveRanks = 0.0;
    if (is_channel) {
        const double ratio = length / height;
        expectedActiveRanks = (dim == 2) ? ratio * n * n : ratio * n * n * n;
    } else if (is_bfs) {
        expectedActiveRanks = (dim == 2) ? min_subdomains * n * n : min_subdomains * n * n * n;
    } else {
        expectedActiveRanks = (dim == 2) ? n * n : n * n * n;
    }

    std::cout << "\n-- Domain stats --" << std::endl;
    std::cout << "  dim                               : " << dim << " (domain dimension)" << std::endl;
    std::cout << "  meshType                          : " << meshType
              << " (structured=channel, structured_ldc=LDC, structured_bfs=BFS;"
              << " channel/LDC use Domain geometry label \"Square\" -> buildMesh2D rectangle)"
              << std::endl;
    std::cout << "  total_mpi_ranks                   : " << total_mpi_ranks << std::endl;
    std::cout << "  mpi_ranks_coarse_solve            : " << mpi_ranks_coarse_solve
              << " (reserved for coarse solve; General / \"Mpi Ranks Coarse\")" << std::endl;
    std::cout << "  active_ranks                      : " << active_ranks
              << " (total_mpi_ranks - mpi_ranks_coarse_solve)" << std::endl;
    std::cout << "  length (CLI)                      : " << length << " (BFS: step length; channel: domain x extent)"
              << std::endl;
    if (is_bfs) {
        std::cout << "  min_subdomains (2*length+1)       : " << min_subdomains << std::endl;
        std::cout << "  bfs_multiplier (2*(length+1)-1)   : " << bfs_multiplier << std::endl;
        std::cout << "  ratio (length / height)           : N/A (not used for BFS)" << std::endl;
    } else {
        std::cout << "  height                            : " << height
                  << " (2D: y extent; 3D: yz-square side, width assumed equal)" << std::endl;
        std::cout << "  ratio (length / height)           : " << (height > 0. ? length / height : 0.)
                  << " (used with structured channel partitioning)" << std::endl;
    }
    std::cout << "  n (MeshStructured subdomain count) : " << n
              << " (channel: Ny=n,Nx from length; LDC: uniform; BFS: n^dim patches per step block)" << std::endl;
    std::cout << "  m (H/h)                           : " << m << " (elements per subdomain patch edge)"
              << std::endl;
    std::cout << "  expected_active_ranks_consistency : " << expectedActiveRanks
              << " (should equal active_ranks)" << std::endl;
    std::cout << "-- end Domain stats --" << std::endl;
}

} // namespace Common
} // namespace Tests
} // namespace Problems
} // namespace FEDD

#endif
