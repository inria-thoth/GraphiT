#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <TH/THBlas.h>
#include <vector>


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
void dpooling_max_forward_worker(
    torch::Tensor output,
    const torch::Tensor input, const torch::Tensor kernel_size,
    const torch::Tensor indices,
    int64_t hidden_size, int64_t size_out) {

    auto kernel_size_accessor = kernel_size.accessor<int64_t, 1>();
    auto output_accessor = output.accessor<scalar_t, 2>();
    auto input_accessor = input.accessor<scalar_t, 2>();
    auto indices_accessor = indices.accessor<int64_t, 2>();

    torch::parallel_for(0, hidden_size, 0, [&](int64_t start, int64_t end) {
        for (auto col = start; col < end; ++col) {
            for (int64_t row = 0; row < size_out; ++row) {
                int64_t s = (row == 0) ? 0 : kernel_size_accessor[row - 1];
                // auto max_val = std::numeric_limits<scalar_t>::lowest();
                scalar_t max_val = 0;
                int64_t max_index = -1;
                for (int64_t k = s; k < kernel_size_accessor[row]; ++k) {
                    auto val = input_accessor[k][col];
                    if (val > max_val) {
                        max_val = val;
                        max_index = k;
                    }
                }
                output_accessor[row][col] = max_val;
                indices_accessor[row][col] = max_index;
            }
        }
    });
}

std::vector<torch::Tensor> dpooling_max_forward_cpu(
    torch::Tensor input,
    torch::Tensor kernel_size) {
    // input: H_in x hidden_size
    // kernel_size: H_out; sum(kernel_size) = H
    // output: H_out x hidden_size

    const int64_t hidden_size = input.size(1);
    const int64_t size_out = kernel_size.size(0);
    auto output = torch::empty({size_out, hidden_size}, input.options());
    auto indices = torch::empty({size_out, hidden_size}, kernel_size.options());

    auto commonDtype = promoteTypes(output.scalar_type(), input.scalar_type());

    // const auto indices_size = kernel_size.cumsum(0);

    AT_DISPATCH_FLOATING_TYPES(
        commonDtype, "dpooling_max_forward", [&] {
            dpooling_max_forward_worker<scalar_t>(output, input, kernel_size, indices, hidden_size, size_out);
        }
    );

    return {output, indices};
}

template <typename scalar_t>
void dpooling_max_backward_worker(
    torch::Tensor d_input,
    const torch::Tensor d_output, const torch::Tensor indices,
    int64_t hidden_size, int64_t size) {

    auto d_output_accessor = d_output.accessor<scalar_t, 2>();
    auto d_input_accessor = d_input.accessor<scalar_t, 2>();
    auto indices_accessor = indices.accessor<int64_t, 2>();

    torch::parallel_for(0, hidden_size, 0, [&](int64_t start, int64_t end) {
        for (auto col = start; col < end; ++col) {
            torch::parallel_for(0, size, 0, [&](int64_t s, int64_t e) {
                for (auto row = s; row < e; ++row) {
                    int64_t input_row = indices_accessor[row][col];
                    if (input_row != -1)
                        d_input_accessor[input_row][col] = d_output_accessor[row][col];
                }
            });
        }
    });
}

void dpooling_max_backward_cpu(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor indices) {

    const auto size = d_output.size(0);
    const auto hidden_size = d_input.size(1);

    auto commonDtype = promoteTypes(d_output.scalar_type(), d_input.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(
        commonDtype, "dpooling_max_backward", [&] {
            dpooling_max_backward_worker<scalar_t>(d_input, d_output, indices, hidden_size, size);
        }
    );
}

template <typename scalar_t>
void dpooling_sum_forward_worker(
    torch::Tensor output,
    const torch::Tensor input, const torch::Tensor kernel_size,
    int64_t hidden_size, int64_t size_out, bool mean) {

    auto kernel_size_accessor = kernel_size.accessor<int64_t, 1>();
    scalar_t* output_ptr = output.data_ptr<scalar_t>();
    scalar_t* input_ptr = input.data_ptr<scalar_t>();

    torch::parallel_for(0, size_out, 0, [&](int64_t start, int64_t end) {
        for (auto row = start; row < end; ++row) {
            int64_t s = (row == 0) ? 0 : kernel_size_accessor[row - 1];
            int64_t e = kernel_size_accessor[row];
            scalar_t val = 1;
            if (mean)
                val = 1. / (e - s);
            for (int64_t k = s; k < e; ++k) {
                THBlas_axpy<scalar_t>(hidden_size, val,
                    input_ptr + k * hidden_size, 1,
                    output_ptr + row * hidden_size, 1);
            }
        }
    });
}

torch::Tensor dpooling_sum_forward_cpu(
    torch::Tensor input,
    torch::Tensor kernel_size,
    bool mean) {
    // input: H_in x hidden_size
    // kernel_size: H_out; sum(kernel_size) = H
    // output: H_out x hidden_size

    const int64_t hidden_size = input.size(1);
    const int64_t size_out = kernel_size.size(0);
    auto output = torch::zeros({size_out, hidden_size}, input.options());

    auto commonDtype = promoteTypes(output.scalar_type(), input.scalar_type());

    // const auto indices_size = kernel_size.cumsum(0);

    AT_DISPATCH_FLOATING_TYPES(
        commonDtype, "dpooling_sum_forward", [&] {
            dpooling_sum_forward_worker<scalar_t>(output, input, kernel_size, hidden_size, size_out, mean);
        }
    );

    return output;
}

template <typename scalar_t>
void dpooling_sum_backward_worker(
    torch::Tensor d_input,
    const torch::Tensor d_output, const torch::Tensor kernel_size,
    int64_t hidden_size, int64_t size_out, bool mean) {

    auto kernel_size_accessor = kernel_size.accessor<int64_t, 1>();
    scalar_t* d_output_ptr = d_output.data_ptr<scalar_t>();
    scalar_t* d_input_ptr = d_input.data_ptr<scalar_t>();

    torch::parallel_for(0, size_out, 0, [&](int64_t start, int64_t end) {
        for (auto row = start; row < end; ++row) {
            int64_t s = (row == 0) ? 0 : kernel_size_accessor[row - 1];
            int64_t e = kernel_size_accessor[row];
            scalar_t val = 1;
            if (mean)
                val = 1. / (e - s);
            for (int64_t k = s; k < e; ++k) {
                THBlas_axpy<scalar_t>(hidden_size, val,
                    d_output_ptr + row * hidden_size, 1,
                    d_input_ptr + k * hidden_size, 1);
            }
        }
    });
}

void dpooling_sum_backward_cpu(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor kernel_size,
    bool mean) {

    const auto size_out = d_output.size(0);
    const auto hidden_size = d_input.size(1);

    auto commonDtype = promoteTypes(d_output.scalar_type(), d_input.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(
        commonDtype, "dpooling_sum_backward", [&] {
            dpooling_sum_backward_worker<scalar_t>(d_input, d_output, kernel_size, hidden_size, size_out, mean);
        }
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_forward", &dpooling_max_forward_cpu, "dynamic max pooling forward (CPU)");
    m.def("max_backward", &dpooling_max_backward_cpu, "dynamic max pooling backward (CPU)");
    m.def("sum_forward", &dpooling_sum_forward_cpu, "dynamic sum/mean pooling forward (CPU)");
    m.def("sum_backward", &dpooling_sum_backward_cpu, "dynamic sum/mean pooling backward (CPU)");
}
