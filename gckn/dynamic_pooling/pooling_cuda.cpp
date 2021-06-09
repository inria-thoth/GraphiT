#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> dpooling_max_cuda_forward(
    torch::Tensor input,
    torch::Tensor kernel_size);

void dpooling_max_cuda_backward(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor indices);

torch::Tensor dpooling_sum_cuda_forward(
    torch::Tensor input,
    torch::Tensor kernel_size,
    bool mean);

void dpooling_sum_cuda_backward(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor kernel_size,
    bool mean);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> dpooling_max_forward(
    torch::Tensor input,
    torch::Tensor kernel_size) {
    CHECK_INPUT(input);
    CHECK_INPUT(kernel_size);

    return dpooling_max_cuda_forward(input, kernel_size);
}

void dpooling_max_backward(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor indices) {
    CHECK_INPUT(d_output);
    CHECK_INPUT(d_input);
    CHECK_INPUT(indices);
    dpooling_max_cuda_backward(d_input, d_output, indices);
}

torch::Tensor dpooling_sum_forward(
    torch::Tensor input,
    torch::Tensor kernel_size,
    bool mean) {
    CHECK_INPUT(input);
    CHECK_INPUT(kernel_size);

    return dpooling_sum_cuda_forward(input, kernel_size, mean);
}

void dpooling_sum_backward(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor kernel_size,
    bool mean) {
    CHECK_INPUT(d_input);
    CHECK_INPUT(d_output);
    CHECK_INPUT(kernel_size);
    dpooling_sum_cuda_backward(d_input, d_output, kernel_size, mean);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_forward", &dpooling_max_forward, "dynamic max pooling forward (CUDA)");
    m.def("max_backward", &dpooling_max_backward, "dynamic max pooling backward (CUDA)");
    m.def("sum_forward", &dpooling_sum_forward, "dynamic sum/mean pooling forward (CUDA)");
    m.def("sum_backward", &dpooling_sum_backward, "dynamic sum/mean pooling backward (CUDA)");
}
