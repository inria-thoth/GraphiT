#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <TH/THBlas.h>


template<typename T>
inline void THBlas_axpy(int64_t n, T a, T *x, int64_t incx, T *y, int64_t incy);

#define AXPY_SPECIALIZATION(ctype,name) \
    template<> \
    inline void THBlas_axpy<ctype>(int64_t n, ctype a, ctype *x, int64_t incx, \
                        ctype *y, int64_t incy) { \
        TH ## name ## Blas_axpy(n, a, x, incx, y, incy); \
    }

AT_FORALL_SCALAR_TYPES(AXPY_SPECIALIZATION)

template <typename scalar_t>
void path_conv_forward_worker(
    torch::Tensor output,
    const torch::Tensor path_indices, const torch::Tensor features,
    int64_t n_paths, int64_t path_size, int64_t feat_path_size, int64_t hidden_size) {

    auto path_indices_accessor = path_indices.accessor<int64_t, 2>();
    scalar_t* features_ptr = features.data_ptr<scalar_t>();
    scalar_t* output_ptr = output.data_ptr<scalar_t>();
    // scalar_t val = 1. / path_size;

    // int64_t i, j;
    // for (i = 0; i < n_paths; ++i) {
    //     for (j = 0; j < path_size; ++j) {
    //         int64_t node_idx = path_indices_accessor[i][j];
    //         THBlas_axpy<scalar_t>(hidden_size, val,
    //             features_ptr + node_idx * path_size * hidden_size + j * hidden_size, 1,
    //             output_ptr + i * hidden_size, 1);
    //     }
    // }
    torch::parallel_for(0, n_paths, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
            for (int64_t j = 0; j < path_size; j++) {
                int64_t node_idx = path_indices_accessor[i][j];
                THBlas_axpy<scalar_t>(hidden_size, 1.,
                    features_ptr + (node_idx * feat_path_size + j) * hidden_size, 1,
                    output_ptr + i * hidden_size, 1);
            }
        }
    });
}

torch::Tensor path_conv_forward_cpu(
    torch::Tensor path_indices,
    torch::Tensor features) {
    // path_indices: n_paths x path_size (value < n_nodes)
    // features: n_nodes x path_size x hidden_size x (in_path_size)
    // output: n_paths x hidden_size x (in_path_size)
    const int64_t n_paths = path_indices.size(0);
    const int64_t path_size = path_indices.size(1);
    const int64_t feat_path_size = features.size(1); // should be >= path_size
    const int64_t hidden_size = features.size(2);

    auto output = torch::zeros({n_paths, hidden_size}, features.options());
    auto commonDtype = promoteTypes(features.scalar_type(), output.scalar_type());

    AT_DISPATCH_ALL_TYPES(
        commonDtype, "path_conv_forward", [&] {
            path_conv_forward_worker<scalar_t>(output, path_indices, features, n_paths, path_size, feat_path_size, hidden_size);
        }
    );

    output /= path_size;

    return output;
}

template <typename scalar_t>
void path_conv_backward_worker(
    torch::Tensor d_input,
    const torch::Tensor path_indices, const torch::Tensor d_output,
    int64_t n_paths, int64_t path_size, int64_t feat_path_size, int64_t hidden_size) {

    auto path_indices_accessor = path_indices.accessor<int64_t, 2>();
    scalar_t* d_input_ptr = d_input.data_ptr<scalar_t>();
    scalar_t* d_output_ptr = d_output.data_ptr<scalar_t>();
    // scalar_t val = 1. / path_size;

    torch::parallel_for(0, n_paths, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; ++i) {
            for (int64_t j = 0; j < path_size; ++j) {
                int64_t node_idx = path_indices_accessor[i][j];
                THBlas_axpy<scalar_t>(hidden_size, 1.,
                    d_output_ptr + i * hidden_size, 1,
                    d_input_ptr + (node_idx * feat_path_size + j) * hidden_size, 1);
            }
        }
    });
}

void path_conv_backward_cpu(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor path_indices) {
    const int64_t n_paths = path_indices.size(0);
    const int64_t path_size = path_indices.size(1);
    const int64_t feat_path_size = d_input.size(1);
    const int64_t hidden_size = d_output.size(1);
    // auto d_input = torch::zeros({n_nodes, path_size, hidden_size}, d_output.options());

    auto commonDtype = promoteTypes(d_input.scalar_type(), d_output.scalar_type());

    AT_DISPATCH_ALL_TYPES(
        commonDtype, "path_conv_backward", [&] {
            path_conv_backward_worker<scalar_t>(d_input, path_indices, d_output, n_paths, path_size, feat_path_size, hidden_size);
        }
    );
    d_input /= path_size;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("path_conv_forward", &path_conv_forward_cpu, "path kernel mapping forward (CPU)");
    m.def("path_conv_backward", &path_conv_backward_cpu, "path kernel mapping backward (CPU)");
}
