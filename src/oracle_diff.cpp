// oracle_diff.cpp — diff two oracle dumps for strict semantic equivalence.
//
// Usage:
//   oracle_diff <ref_dump> <other_dump> [--tol 1e-4]
//
// Exit code: 0 if all checkpoints match within tol; 1 otherwise.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "oracle.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <ref_dump> <other_dump> [--tol N]\n", argv[0]);
        return 2;
    }
    std::string ref = argv[1];
    std::string oth = argv[2];
    double tol = 1e-4;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tol") && i+1 < argc) {
            tol = std::atof(argv[++i]);
        }
    }
    auto r = deploy::diff_dumps(ref, oth, tol);
    std::fprintf(stdout, "ref      : %s\n", ref.c_str());
    std::fprintf(stdout, "other    : %s\n", oth.c_str());
    std::fprintf(stdout, "checkpts : %zu\n", r.n_cp);
    std::fprintf(stdout, "shape ok : %s\n", r.ok ? "yes" : "NO");
    std::fprintf(stdout, "max |Δ|  : %.6e\n", r.max_abs_delta);
    std::fprintf(stdout, "mismatch : %zu / %zu  (tol = %.2e)\n",
                 r.n_mismatched_cp, r.n_cp, tol);
    return (r.ok && r.n_mismatched_cp == 0) ? 0 : 1;
}
