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
        std::int64_t a1 = whf[1];
        std::int64_t a2 = whf[2];
        std::int64_t a3 = whf[3];
        std::int64_t sum = a1 + a2 + a3;
        std::int64_t diff = whf[0] - sum;
        whf[0] += sum;
        whf[1] = diff + (a2 << 1);
        whf[2] = diff + (a1 << 1);
        whf[3] = diff + (a3 << 1);
    }

    if (n > 2) {
        std::size_t shift0 = ((((n%3)^3)%3) << 1); // n%3 = {0, 1, 2} -> {0, 4, 2}

        std::size_t k = n - 2;
        std::size_t two_to_k = (1 << k);
        for (size_t shift = 0; shift < shift0; shift += 2) {
            std::size_t jump1 = (1 << shift);
            std::size_t jump2 = (2 << shift);
            std::size_t jump3 = (3 << shift);
            #pragma omp parallel for
            for (size_t idx = 0; idx < two_to_k; ++idx) {
                std::size_t phase = (idx >> shift)&3;
                std::size_t loc_a0 = idx ^ (phase << shift) ^ (phase << k);
                std::size_t loc_a1 = loc_a0 ^ jump1;
                std::size_t loc_a2 = loc_a0 ^ jump2;
                std::size_t loc_a3 = loc_a0 ^ jump3;
                std::int64_t a1 = whf[loc_a1];
                std::int64_t a2 = whf[loc_a2];
                std::int64_t a3 = whf[loc_a3];
                std::int64_t sum = a1 + a2 + a3;
                std::int64_t diff = whf[loc_a0] - sum;
                whf[loc_a0] += sum;
                whf[loc_a1] = diff + (a2 << 1);
                whf[loc_a2] = diff + (a1 << 1);
                whf[loc_a3] = diff + (a3 << 1);
            }
        }

        k = n - 3;
        two_to_k = (1 << k);
        for (std::size_t shift = shift0; shift < n; shift += 3) {
            std::size_t jump1 = (1 << shift);
            std::size_t jump2 = (2 << shift);
            std::size_t jump3 = (3 << shift);
            std::size_t jump4 = (4 << shift);
            std::size_t jump5 = (5 << shift);
            std::size_t jump6 = (6 << shift);
            std::size_t jump7 = (7 << shift);
            #pragma omp parallel for
            for (size_t idx = 0; idx < two_to_k; ++idx) {
                std::size_t phase = (idx >> shift)&7;
                std::size_t loc_a0 = idx ^ (phase << shift) ^ (phase << k);
                std::size_t loc_a1 = loc_a0 ^ jump1;
                std::size_t loc_a2 = loc_a0 ^ jump2;
                std::size_t loc_a3 = loc_a0 ^ jump3;
                std::size_t loc_a4 = loc_a0 ^ jump4;
                std::size_t loc_a5 = loc_a0 ^ jump5;
                std::size_t loc_a6 = loc_a0 ^ jump6;
                std::size_t loc_a7 = loc_a0 ^ jump7;
                std::int64_t a1 = whf[loc_a1] << 1;
                std::int64_t a2 = whf[loc_a2] << 1;
                std::int64_t a3 = whf[loc_a3] << 1;
                std::int64_t a4 = whf[loc_a4] << 1;
                std::int64_t a5 = whf[loc_a5] << 1;
                std::int64_t a6 = whf[loc_a6] << 1;
                std::int64_t a7 = whf[loc_a7] << 1;
                std::int64_t b0 = a1 + a2;
                std::int64_t b1 = a3 + a7;
                std::int64_t b2 = a5 + a6;
                std::int64_t sum = (b0 + b1 + b2 + a4) >> 1;
                std::int64_t diff = whf[loc_a0] - sum;
                std::int64_t c0 = diff + a3;
                std::int64_t c1 = diff + a4;
                std::int64_t c2 = diff + a7;
                whf[loc_a0] += sum;
                whf[loc_a1] = c1 + a2 + a6;
                whf[loc_a2] = c1 + a1 + a5;
                whf[loc_a3] = c1 + b1;
                whf[loc_a4] = c0 + b0;
                whf[loc_a5] = c2 + a2 + a5;
                whf[loc_a6] = c2 + a1 + a6;
                whf[loc_a7] = c0 + b2;
            }
        }
    }

    return whf;
}