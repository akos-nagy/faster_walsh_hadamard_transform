#include <vector>
#include <omp.h>
#include <stdexcept>
#include <cstdint>

std::vector<int64_t> fasterWalshHadamardTransform(const std::vector<int64_t>& f) {
    size_t N = f.size();
    if (__builtin_popcount(N) != 1) {
        throw std::runtime_error("Input length must be a power of 2.");
    }

    std::vector<int64_t> whf = f;

    std::size_t n = __builtin_ctz(N);

    if (n == 1) {
        whf[0] += whf[1];
        whf[1] = whf[0] - (whf[1] << 1);
    }

    if (n == 2) {
        std::int64_t b = whf[1];
        std::int64_t c = whf[2];
        std::int64_t d = whf[3];
        std::int64_t s = b + c + d;
        std::int64_t t = whf[0] - s;
        whf[0] += s;
        whf[1] = t + (c << 1);
        whf[2] = t + (b << 1);
        whf[3] = t + (d << 1);
    }

    if (n > 2) {
        std::size_t k = n - 2;
        std::size_t L = (1 << k);
        std::size_t shift0 = 0;
        if (n%3 == 1) {
            shift0 = 4;
        }
        if (n%3 == 2) {
            shift0 = 2;
        }

        for (size_t shift = 0; shift < shift0; shift += 2) {
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

        k = n - 3;
        L = (1 << k);
        for (std::size_t shift = shift0; shift < n; shift += 3) {
            #pragma omp parallel for
            for (size_t idx = 0; idx < L; ++idx) {
                std::size_t phase = (idx >> shift)&7;
                std::size_t loc_a0 = idx ^ (phase << shift) ^ (phase << k);
                std::size_t loc_a1 = loc_a0 ^ (1 << shift);
                std::size_t loc_a2 = loc_a0 ^ (2 << shift);
                std::size_t loc_a3 = loc_a0 ^ (3 << shift);
                std::size_t loc_a4 = loc_a0 ^ (4 << shift);
                std::size_t loc_a5 = loc_a0 ^ (5 << shift);
                std::size_t loc_a6 = loc_a0 ^ (6 << shift);
                std::size_t loc_a7 = loc_a0 ^ (7 << shift);
                std::int64_t a1 = whf[loc_a1];
                std::int64_t a2 = whf[loc_a2];
                std::int64_t a3 = whf[loc_a3];
                std::int64_t a4 = whf[loc_a4];
                std::int64_t a5 = whf[loc_a5];
                std::int64_t a6 = whf[loc_a6];
                std::int64_t a7 = whf[loc_a7];
                std::int64_t b0 = a1 + a2;
                std::int64_t b1 = a3 + a7;
                std::int64_t b2 = a5 + a6;
                std::int64_t s = b0 + b1 + b2 + a4;
                std::int64_t t = whf[loc_a0] - s;
                std::int64_t c0 = t + (a3 << 1);
                std::int64_t c1 = t + (a4 << 1);
                std::int64_t c2 = t + (a7 << 1);
                whf[loc_a0] += s;
                whf[loc_a1] = c1 + ((a2 + a6) << 1);
                whf[loc_a2] = c1 + ((a1 + a5) << 1);
                whf[loc_a3] = c1 + (b1 << 1);
                whf[loc_a4] = c0 + (b0 << 1);
                whf[loc_a5] = c2 + ((a2 + a5) << 1);
                whf[loc_a6] = c2 + ((a1 + a6) << 1);
                whf[loc_a7] = c0 + (b2 << 1);
            }
        }
    }

    return whf;
}