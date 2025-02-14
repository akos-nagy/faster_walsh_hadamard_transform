#include <vector>
#include <stdexcept>
#include <omp.h>
#include <cstdint>

std::vector<int64_t> fasterWalshHadamardTransform(const std::vector<int64_t>& f) {
    size_t N = f.size();
    if (__builtin_popcount(N) != 1) {
        throw std::runtime_error("Input length must be a power of 2.");
    }

    std::vector<int64_t> whf = f;

    std::size_t n = __builtin_ctz(N);
    std::size_t parity = n%2;
    std::size_t k = n - 2;
    std::size_t L = (N >> 2);

    if (parity) {
        if (n == 1) {
            whf[0] += whf[1];
            whf[1] = whf[0] - (whf[1] << 1);
        } else {
            #pragma omp parallel for
            for (size_t idx = 0; idx < N; idx += 8) {
                std::int64_t a1 = whf[idx ^ 1];
                std::int64_t a2 = whf[idx ^ 2];
                std::int64_t a3 = whf[idx ^ 3];
                std::int64_t a4 = whf[idx ^ 4];
                std::int64_t a5 = whf[idx ^ 5];
                std::int64_t a6 = whf[idx ^ 6];
                std::int64_t a7 = whf[idx ^ 7];
                std::int64_t b0 = a1 + a2;
                std::int64_t b1 = a3 + a7;
                std::int64_t b2 = a5 + a6;
                std::int64_t s = b0 + b1 + b2 + a4;
                std::int64_t t = whf[idx] - s;
                std::int64_t c0 = t + (a3 << 1);
                std::int64_t c1 = t + (a4 << 1);
                std::int64_t c2 = t + (a7 << 1);
                whf[idx] += s;
                whf[idx ^ 1] = c1 + ((a2 + a6) << 1);
                whf[idx ^ 2] = c1 + ((a1 + a5) << 1);
                whf[idx ^ 3] = c1 + (b1 << 1);
                whf[idx ^ 4] = c0 + (b0 << 1);
                whf[idx ^ 5] = c2 + ((a2 + a5) << 1);
                whf[idx ^ 6] = c2 + ((a1 + a6) << 1);
                whf[idx ^ 7] = c0 + (b2 << 1);
            }
        }
    }

    if (n > 1) {
        for (std::size_t shift = ((parity) ? 3 : 0); shift <= k; shift += 2) {
            #pragma omp parallel for
            for (size_t idx = 0; idx < L; ++idx) {
                std::size_t phase = (idx >> shift)&3;
                std::size_t loc_a = idx ^ (phase << shift) ^ (phase << k);
                std::size_t loc_b = loc_a ^ (1 << shift);
                std::size_t loc_c = loc_a ^ (2 << shift);
                std::size_t loc_d = loc_c ^ (1 << shift);
                std::int64_t b = whf[loc_b];
                std::int64_t c = whf[loc_c];
                std::int64_t d = whf[loc_d];
                std::int64_t s = b + c + d;
                std::int64_t t = whf[loc_a] - s;
                whf[loc_a] += s;
                whf[loc_b] = t + (c << 1);
                whf[loc_c] = t + (b << 1);
                whf[loc_d] = t + (d << 1);
            }
        }
    }
    }

    return whf;
}