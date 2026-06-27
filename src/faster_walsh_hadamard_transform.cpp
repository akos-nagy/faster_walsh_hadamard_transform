#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <omp.h>

std::vector<int64_t> fasterWalshHadamardTransform(const std::vector<int64_t>& f) {
    const std::size_t N = f.size();
    if (!std::has_single_bit(N)) {
        throw std::runtime_error("Input length must be a positive power of 2.");
    }

    if (N == 1) {
        return f;
    }

    std::vector<std::int64_t> whf = f;

    const unsigned n = static_cast<unsigned>(std::countr_zero(N));

    if (n > 2) {
        constexpr unsigned shift0_table[3] = {0, 4, 2};
        const unsigned shift0 = shift0_table[n % 3]; // n%3 = {0, 1, 2} -> {0, 4, 2}

        const unsigned k2 = n - 2;
        const std::size_t two_to_k2 = N >> 2;
        for (size_t shift = 0; shift < shift0; shift += 2) {
            std::size_t jump1 = std::size_t{1} << shift;
            std::size_t jump2 = std::size_t{2} << shift;
            std::size_t jump3 = std::size_t{3} << shift;
            #pragma omp parallel for schedule(static)
            for (size_t idx = 0; idx < two_to_k2; ++idx) {
                std::size_t phase = (idx >> shift)&3;
                std::size_t loc_a0 = idx ^ (phase << shift) ^ (phase << k2);
                std::size_t loc_a1 = loc_a0 ^ jump1;
                std::size_t loc_a2 = loc_a0 ^ jump2;
                std::size_t loc_a3 = loc_a0 ^ jump3;
                std::int64_t a1 = whf[loc_a1];
                std::int64_t a2 = whf[loc_a2];
                std::int64_t a3 = whf[loc_a3];
                std::int64_t sum = a1 + a2 + a3;
                std::int64_t diff = whf[loc_a0] - sum;
                whf[loc_a0] += sum;
                whf[loc_a1] = diff + 2 * a2;
                whf[loc_a2] = diff + 2 * a1;
                whf[loc_a3] = diff + 2 * a3;
            }
        }

        const unsigned k3 = n - 3;
        const std::size_t two_to_k3 = N >> 3;
        for (std::size_t shift = shift0; shift < n; shift += 3) {
            std::size_t jump1 = std::size_t{1} << shift;
            std::size_t jump2 = std::size_t{2} << shift;
            std::size_t jump3 = std::size_t{3} << shift;
            std::size_t jump4 = std::size_t{4} << shift;
            std::size_t jump5 = std::size_t{5} << shift;
            std::size_t jump6 = std::size_t{6} << shift;
            std::size_t jump7 = std::size_t{7} << shift;
            #pragma omp parallel for schedule(static)
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
                std::int64_t a1 = 2 * whf[loc_a1];
                std::int64_t a2 = 2 * whf[loc_a2];
                std::int64_t a3 = 2 * whf[loc_a3];
                std::int64_t a4 = 2 * whf[loc_a4];
                std::int64_t a5 = 2 * whf[loc_a5];
                std::int64_t a6 = 2 * whf[loc_a6];
                std::int64_t a7 = 2 * whf[loc_a7];
                std::int64_t b0 = a1 + a2;
                std::int64_t b1 = a3 + a7;
                std::int64_t b2 = a5 + a6;
                std::int64_t sum = (b0 + b1 + b2 + a4) / 2;
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
    } else if (n == 2) { // the n == 2 trick
        std::int64_t a1 = whf[1];
        std::int64_t a2 = whf[2];
        std::int64_t a3 = whf[3];
        std::int64_t sum = a1 + a2 + a3;
        std::int64_t diff = whf[0] - sum;
        whf[0] += sum;
        whf[1] = diff + 2 * a2;
        whf[2] = diff + 2 * a1;
        whf[3] = diff + 2 * a3;
    } else { // at this point, n is guaranteed to be a positive integer that is less than 2, thus n == 1
        const std::int64_t a0 = whf[0];
        const std::int64_t a1 = whf[1];

        whf[0] = a0 + a1;
        whf[1] = a0 - a1;
    }

    return whf;
}