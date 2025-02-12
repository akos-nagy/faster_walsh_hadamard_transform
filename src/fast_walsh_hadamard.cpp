#include <vector>
#include <stdexcept>
#include <omp.h>
#include <cstdint>

std::vector<int64_t> fastWalshHadamardTransform(const std::vector<int64_t>& f) {
    size_t N = f.size();
    if (__builtin_popcount(N) != 1) {
        throw std::runtime_error("Input length must be a power of 2.");
    }

    std::vector<int64_t> result = f;  // Copy input to avoid in-place modification
    size_t M = (N >> 1);
    for (size_t shift = 1; shift <= M; shift <<= 1) {
        size_t dual = shift ^ M;
        #pragma omp parallel for
        for (size_t idx = 0; idx < M; ++idx) {
            size_t loc = idx ^ ((idx & shift) ? dual : 0);
            int64_t a = result[loc];
            int64_t b = result[loc ^ shift];
            result[loc] += b;
            result[loc ^ shift] = a - b;
        }
    }
    return result;
}