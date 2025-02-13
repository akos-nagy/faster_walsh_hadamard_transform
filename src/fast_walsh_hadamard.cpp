#include <vector>
#include <stdexcept>
#include <omp.h>
#include <cstdint>

std::vector<int64_t> fastWalshHadamardTransform(const std::vector<int64_t>& f) {
    size_t N = f.size();
    if (__builtin_popcount(N) != 1) {
        throw std::runtime_error("Input length must be a power of 2.");
    }

    std::size_t n = __builtin_ctz(N);
    std::size_t parity = n&1;
    std::size_t L = (N >> 2);

    std::size_t M = (N >> 1);
    if (parity == 1) {
        #pragma omp parallel for
        for (size_t idx = 0; idx < M; ++idx) {
            std::int64_t a = f[idx << 1];
            std::int64_t b = f[(idx << 1) ^ 1];
            f[idx << 1] = a + b;
            f[(idx << 1) ^ 1] = a - b;
        }
    }

    if (n > 1) {
        for (std::size_t shift = parity; shift < n - 1; shift += 2) {
            #pragma omp parallel for
            for (size_t idx = 0; idx < L; ++idx) {
                std::size_t phase = (idx >> shift)&3; // 0, 1, 2, or 3
                std::size_t jump = phase * L;
                std::size_t loc_a = idx ^ (phase << shift) ^ jump;
                std::size_t loc_b = loc_a ^ (1 << shift);
                std::size_t loc_c = loc_a ^ (2 << shift);
                std::size_t loc_d = loc_c ^ (1 << shift);
                std::int64_t b = f[loc_b];
                std::int64_t c = f[loc_c];
                std::int64_t d = f[loc_d];
                std::int64_t s = b + c + d;
                std::int64_t t = f[loc_a] - s;
                f[loc_a] += s;
                f[loc_b] = t + (c << 1);
                f[loc_c] = t + (b << 1);
                f[loc_d] = t + (d << 1);
            }
        }
    }

    return result;
}