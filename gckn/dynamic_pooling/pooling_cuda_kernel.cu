#include <torch/extension.h>

#include <vector>


namespace {
template <typename scalar_t>
__global__ void dpooling_max_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ kernel_size,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    size_t hidden_size) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int index = row * hidden_size + column;

    const int64_t start = (row == 0) ? 0 : kernel_size[row - 1];
    int64_t curr_index;
    scalar_t val;
    scalar_t max_val = 0;
    int64_t max_index = -1;

    if (column < hidden_size) {
        for (int64_t k = start; k < kernel_size[row]; ++k) {
            curr_index = k * hidden_size + column;
            val = input[curr_index];
            if (val > max_val) {
                max_val = val;
                max_index = k;
            }
        }
        output[index] = max_val;
        indices[index] = max_index;
    }
}

template <typename scalar_t>
__global__ void dpooling_max_cuda_backward_kernel(
    const scalar_t* __restrict__ d_output,
    const int64_t* __restrict__ indices,
    scalar_t* __restrict__ d_input,
    size_t hidden_size) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int index = row * hidden_size + column;
    const int64_t index_row = indices[index];

    if (column < hidden_size) {
        if (index_row != -1)
            d_input[index_row * hidden_size + column] = d_output[index];
    }
}

template <typename scalar_t>
__global__ void dpooling_sum_cuda_forward_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ kernel_size,
    int64_t hidden_size, bool mean) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int index = row * hidden_size + column;

    const int64_t start = (row == 0) ? 0 : kernel_size[row - 1];
    const int64_t end = kernel_size[row];
    int64_t curr_index = start * hidden_size + column;
    scalar_t val = 1;
    if (mean)
        val = 1. / (end - start);

    if (column < hidden_size) {
        for (int64_t k = start; k < end; ++k) {
            output[index] += val * input[curr_index];
            curr_index += hidden_size;
        }
    }
}

template <typename scalar_t>
__global__ void dpooling_sum_cuda_backward_kernel(
    scalar_t* __restrict__ d_input,
    const scalar_t* __restrict__ d_output,
    const int64_t* __restrict__ kernel_size,
    int64_t hidden_size, bool mean) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int index = row * hidden_size + column;

    const int64_t start = (row == 0) ? 0 : kernel_size[row - 1];
    const int64_t end = kernel_size[row];
    int64_t curr_index = start * hidden_size + column;
    scalar_t val = 1;
    if (mean)
        val = 1. / (end - start);

    if (column < hidden_size) {
        for (int64_t k = start; k < end; ++k) {
            d_input[curr_index] += val * d_output[index];
            curr_index += hidden_size;
        }
    }
}

}

std::vector<torch::Tensor> dpooling_max_cuda_forward(
    torch::Tensor input,
    torch::Tensor kernel_size) {
    // input: H_in x hidden_size
    // kernel_size: H_out; sum(kernel_size) = H
    // output: H_out x hidden_size

    const auto size = input.size(0);
    const auto hidden_size = input.size(1);
    const auto size_out = kernel_size.size(0);
    auto output = torch::zeros({size_out, hidden_size}, input.options());
    auto indices = torch::zeros({size_out, hidden_size}, kernel_size.options());

    // const auto indices_size = kernel_size.cumsum(0);

    const int threads = 1024;
    const dim3 blocks((hidden_size + threads - 1) / threads, size_out);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "dpooling_max_forward_cuda", ([&] {
        dpooling_max_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            kernel_size.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            hidden_size);
    }));

    return {output, indices};
}

void dpooling_max_cuda_backward(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor indices) {

    const auto size = d_output.size(0);
    const auto hidden_size = d_input.size(1);

    const int threads = 1024;
    const dim3 blocks((hidden_size + threads - 1) / threads, size);

    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "dpooling_max_backward_cuda", ([&] {
        dpooling_max_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            d_output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            d_input.data_ptr<scalar_t>(),
            hidden_size);
    }));
}

torch::Tensor dpooling_sum_cuda_forward(
    torch::Tensor input,
    torch::Tensor kernel_size,
    bool mean) {
    // input: H_in x hidden_size
    // kernel_size: H_out; sum(kernel_size) = H
    // output: H_out x hidden_size
    const int64_t hidden_size = input.size(1);
    const int64_t size_out = kernel_size.size(0);
    auto output = torch::zeros({size_out, hidden_size}, input.options());

    const int threads = 1024;
    const dim3 blocks((hidden_size + threads - 1) / threads, size_out);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "dpooling_sum_forward_cuda", ([&] {
        dpooling_sum_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            kernel_size.data_ptr<int64_t>(),
            hidden_size,
            mean);
    }));

    return output;
}

void dpooling_sum_cuda_backward(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor kernel_size,
    bool mean) {
    const auto size_out = d_output.size(0);
    const auto hidden_size = d_input.size(1);

    const int threads = 1024;
    const dim3 blocks((hidden_size + threads - 1) / threads, size_out);

    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "dpooling_sum_forward_cuda", ([&] {
        dpooling_sum_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            d_input.data_ptr<scalar_t>(),
            d_output.data_ptr<scalar_t>(),
            kernel_size.data_ptr<int64_t>(),
            hidden_size,
            mean);
    }));
}

