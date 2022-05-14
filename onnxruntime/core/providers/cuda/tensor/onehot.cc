// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/onehot.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

// The types should match the cases in scalar_value_generic() below.
const DeleteOnUnloadPtr<std::vector<MLDataType>> depthTypeConstraints = new std::vector<MLDataType> {
    DataTypeImpl::GetTensorType<float>(),
    DataTypeImpl::GetTensorType<double>(),
    DataTypeImpl::GetTensorType<int8_t>(),
    DataTypeImpl::GetTensorType<int16_t>(),
    DataTypeImpl::GetTensorType<int32_t>(),
    DataTypeImpl::GetTensorType<int64_t>(),
    DataTypeImpl::GetTensorType<uint8_t>(),
    DataTypeImpl::GetTensorType<uint16_t>(),
    DataTypeImpl::GetTensorType<uint32_t>(),
    DataTypeImpl::GetTensorType<uint64_t>()
};

// T1: indices, T2: depth, T3: values
#define REGISTER_TYPED_ONE_HOT_OP(in_type, out_type)                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      OneHot,                                                              \
      kOnnxDomain,                                                         \
      11,                                                                  \
      in_type##_##out_type##_##depth_type,                                 \
      kCudaExecutionProvider,                                              \
      (*KernelDefBuilder::Create())                                        \
          .InputMemoryType(OrtMemTypeCPUInput, 1) /* Keep depth in CPU */  \
          .InputMemoryType(OrtMemTypeCPUInput, 2) /* Keep values in CPU */ \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>())    \
          .TypeConstraint("T2", *depthTypeConstraints)                     \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<out_type>()),  \
      OneHotOp<in_type, out_type, depth_type>);

REGISTER_TYPED_ONE_HOT_OP(int64_t, int64_t)
REGISTER_TYPED_ONE_HOT_OP(int64_t, float)
REGISTER_TYPED_ONE_HOT_OP(int32_t, float)
REGISTER_TYPED_ONE_HOT_OP(int64_t, MLFloat16)
REGISTER_TYPED_ONE_HOT_OP(int32_t, MLFloat16)

template <typename dtype>
int64_t scalar_value(const Tensor* scalar_tensor) {
  const auto* data = scalar_tensor->Data<dtype>();
  return static_cast<int64_t>(*data);
}

int64_t scalar_value_generic(const Tensor* scalar_tensor) {
  auto dtype = scalar_tensor->GetElementType();
  // The cases should match the list of types in depthTypeConstraints.
  switch (dtype) {
    case utils::GetONNXTensorElementDataType<float>():
      return scalar_value<float>(scalar_tensor);
    case utils::GetONNXTensorElementDataType<double>():
      return scalar_value<double>(scalar_tensor);
    case utils::GetONNXTensorElementDataType<int8_t>():
      return scalar_value<int8_t>(scalar_tensor);
    case utils::GetONNXTensorElementDataType<int32_t>():
      return scalar_value<int16_t>(scalar_tensor);
    case utils::GetONNXTensorElementDataType<int32_t>():
      return scalar_value<int32_t>(scalar_tensor);
    case utils::GetONNXTensorElementDataType<int64_t>():
      return scalar_value<int64_t>(scalar_tensor);
    case utils::GetONNXTensorElementDataType<uint8_t>():
      return scalar_value<uint8_t>(scalar_tensor);
    case utils::GetONNXTensorElementDataType<uint16_t>():
      return scalar_value<uint16_t>(scalar_tensor);
    case utils::GetONNXTensorElementDataType<uint32_t>():
      return scalar_value<uint32_t>(scalar_tensor);
    case utils::GetONNXTensorElementDataType<uint64_t>():
      return scalar_value<uint64_t>(scalar_tensor);
    default:
      ORT_THROW("Unsupported 'dtype' value: ", dtype);
  }
}

template <typename in_type, typename out_type>
Status OneHotOp<in_type, out_type, depth_type>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<out_type>::MappedType CudaT_Out;

  const Tensor* indices = ctx->Input<Tensor>(0);
  const Tensor* depth = ctx->Input<Tensor>(1);
  const Tensor* values = ctx->Input<Tensor>(2);

  ORT_RETURN_IF_ERROR(ValidateInputs(depth, values));

  // As per spec in case 'depth' is of non-integer type, it will be casted to int64 before use.
  const int64_t depth_val = scalar_value_generic(depth);
  if (depth_val <= 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Depth is negative.");
  }

  // prepare output shape
  int64_t prefix_dim_size, suffix_dim_size;
  TensorShapeVector output_shape;
  ORT_RETURN_IF_ERROR(PrepareOutputShape(indices, depth_val, axis_, prefix_dim_size, suffix_dim_size, output_shape));

  // allocate output
  const auto* values_data = reinterpret_cast<const CudaT_Out*>(values->Data<out_type>());
  Tensor* output = ctx->Output(0, TensorShape(output_shape));

  // edge case where we have a dim with a value of 0
  if (output->Shape().Size() == 0)
    return Status::OK();

  const fast_divmod fdm_suffix(gsl::narrow_cast<int>(suffix_dim_size));
  const auto* indices_data = indices->Data<in_type>();
  auto* output_data = reinterpret_cast<CudaT_Out*>(output->MutableData<out_type>());

  if (values_data[0] == CudaT_Out(0.f)) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output->MutableDataRaw(), 0, output->SizeInBytes(), Stream()));
    OneHotWithZeroOffValueImpl(Stream(),
                               indices_data,
                               fdm_suffix,
                               depth_val,
                               values_data[1],
                               output_data,
                               indices->Shape().Size());
    return Status::OK();
  }

  const fast_divmod fdm_depth_suffix(gsl::narrow_cast<int>(depth_val * suffix_dim_size));
  OneHotImpl(Stream(),
             indices_data, fdm_depth_suffix, fdm_suffix, depth_val,
             values_data[1],
             values_data[0],
             output_data,
             output->Shape().Size());

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
