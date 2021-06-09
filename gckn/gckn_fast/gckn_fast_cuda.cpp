#include <torch/extension.h>

// CUDA declarations

torch::Tensor path_conv_forward_cuda(
    torch::Tensor path_indices,
    torch::Tensor features);

void path_conv_backward_cuda(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor path_indices);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor path_conv_forward(
    torch::Tensor path_indices,
    torch::Tensor features) {
    CHECK_INPUT(path_indices);
    CHECK_INPUT(features);

    return path_conv_forward_cuda(path_indices, features);
}

void path_conv_backward(
    torch::Tensor d_input,
    torch::Tensor d_output,
    torch::Tensor path_indices) {
    CHECK_INPUT(path_indices);
    CHECK_INPUT(d_input);
    CHECK_INPUT(d_output);

    path_conv_backward_cuda(d_input, d_output, path_indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("path_conv_forward", &path_conv_forward, "path kernel mapping forward (CUDA)");
    m.def("path_conv_backward", &path_conv_backward, "path kernel mapping backward (CUDA)");
}
