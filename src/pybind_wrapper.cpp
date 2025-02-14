#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "faster_walsh_hadamard_transform.cpp"

namespace py = pybind11;

py::array_t<int64_t> fwht_wrapper(const py::array_t<int64_t>& input_array) {
    // Ensure the input is a 1D NumPy array
    py::buffer_info buf = input_array.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }

    // Convert input NumPy array to std::vector<int64_t>
    std::vector<int64_t> input_vec(static_cast<int64_t*>(buf.ptr), 
                                   static_cast<int64_t*>(buf.ptr) + buf.shape[0]);

    // Compute the FWHT
    std::vector<int64_t> result_vec = fasterWalshHadamardTransform(input_vec);

    // Convert result back to a NumPy array
    return py::array_t<int64_t>(result_vec.size(), result_vec.data());
}

PYBIND11_MODULE(faster_walsh_hadamard_transform, m) {
    m.def("fwht", &fwht_wrapper, "Compute the Fast(er) Walsh-Hadamard Transform and return a NumPy array");
}
