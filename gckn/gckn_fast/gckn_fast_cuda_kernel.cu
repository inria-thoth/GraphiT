#include <torch/extension.h>

namespace {
template <typename scalar_t>
__global__ void path_conv_forward_cuda_kernel(
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ path_indices,
    const scalar_t* __restrict__ features,
    int64_t n_paths,
    int64_t path_size,
    int64_t feat_path_size,
    int64_t hidden_size) {

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y;
    const int index = row * hidden_size + col;
    scalar_t val = 1. / path_size;
    int64_t node_idx;

    if (col < hidden_size && row < n_paths) {
        for (int64_t j = 0; j < path_size; ++j){
            node_idx = path_indices[row * path_size + j];
            output[index] += val * features[(node_idx * feat_path_size + j) * hidden_size + col];
        }
    }

}

template <typename scalar_t>
__global__ void path_conv_backward_cuda_kernel(
    scalar_t* __restrict__ d_input,
    const int64_t* __restrict__ path_indices,
    const scalar_t* __restrict__ d_output,
    int64_t n_paths,
    int64_t path_size,
    int64_t feat_path_size,
    int64_t hidden_size) {

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y;
    const int index = row * hidden_size + col;
    scalar_t val = 1. / path_size;
    int64_t node_idx;

    if (col < hidden_size && row < n_paths) {
        for (int64_t j = 0; j < path_size; ++j){
            node_idx = path_indices[row * path_size + j];
            d_input[(node_idx * feat_path_size + j) * hidden_size + col] += val * d_output[index];
        }
    }

}

}

torch::Tensor path_conv_forward_cuda(
    torch::Tensor path_indices,
    torch::Tensor features) {
    // path_indices: n_paths x path_size (value < n_nodes)
    // features: n_nodes x path_size x hidden_size x (in_path_size)
    // output: n_paths x hidden_size x (in_path_size)
    const int64_t n_paths = path_indices.size(0);
    const int64_t path_size = path_indices.size(1);
    const int64_t feat_path_size = features.size(1);
    const int64_t hidden_size = features.size(2);

    auto output = torch::zeros({n_paths, hidden_size}, features.options());

    const int threads = 1024;
    const dim3 blocks((n_paths + threads - 1) / threads, hidden_size);

    AT_DISPATCH_FLOATING_TYPES(features.type(), "path_conv_forward_cuda", ([&] {
        path_conv_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            path_indices.data_ptr<int64_t>(),
            features.data_ptr<scalar_t>(),
            n_paths,
            path_size,
            feat_path_size,
            hidden_size);
    }));

    return output;
}

void path_conv_backward_cuda(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor path_indices) {
    const int64_t n_paths = path_indices.size(0);
    const int64_t path_size = path_indices.size(1);
    const int64_t feat_path_size = d_input.size(1);
    const int64_t hidden_size = d_output.size(1);

    // auto commonDtype = promoteTypes(d_input.scalar_type(), d_output.scalar_type());

    const int threads = 1024;
    const dim3 blocks((n_paths + threads - 1) / threads, hidden_size);

    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "path_conv_backward_cuda", ([&] {
        path_conv_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            d_input.data_ptr<scalar_t>(),
            path_indices.data_ptr<int64_t>(),
            d_output.data_ptr<scalar_t>(),
            n_paths,
            path_size,
            feat_path_size,
            hidden_size);
    }));
}

