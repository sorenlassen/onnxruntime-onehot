/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Modifications Copyright (c) Microsoft. */

#include "core/providers/cpu/tensor/onehot.h"

#include <functional>
#include <limits>
#include <numeric>

#include "core/common/eigen_common_wrapper.h"
#include "core/framework/element_type_lists.h"
#include "core/platform/env.h"
#include "core/providers/common.h"
#include "core/providers/op_kernel_type_control.h"

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif
using namespace ::onnxruntime::common;

namespace onnxruntime {
// spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md#OneHot

// TODO: determine if list should include MLFloat16 and/or BFloat16, or if it's a worthwhile optimization to omit some other types
using AllNumericExceptHalf =
    TypeList<
        float,
        double,
        int64_t,
        uint64_t,
        int32_t,
        uint32_t,
        int16_t,
        uint16_t,
        int8_t,
        uint8_t>;

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 0,
    AllNumericExceptHalf);

// TODO: determine if we should skip this step, or which set of types to require
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 0,
    float, int32_t, int64_t);


ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 1,
    AllNumericExceptHalf);

// TODO: determine if we should skip this step, or which set of types to require
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 1,
    float, int32_t, int64_t);


// TODO: determine whether to omit ML/BFloat16 or other types
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 2,
    element_type_lists::All);

// TODO: determine if we should skip this step, or which set of types to require
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 2,
    float, int32_t, int64_t, std::string);
}  // namespace op_kernel_type_control

namespace {
using IndicesTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 0);
using EnabledIndicesTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 0);

using DepthTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 1);
using EnabledDepthTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 1);

using ValuesTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 2);
using EnabledValuesTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, OneHot, Input, 2);
}  // namespace

// T1: indices, T2: depth, T3: values
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
      OneHot,
      9, 10,
      KernelDefBuilder()
          .TypeConstraint(
              "T1",
              BuildKernelDefConstraintsFromTypeList<IndicesTypes>(),
              BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>())
          .TypeConstraint(
              "T2",
              BuildKernelDefConstraintsFromTypeList<DepthTypes>(),
              BuildKernelDefConstraintsFromTypeList<EnabledDepthTypes>())
          .TypeConstraint(
              "T3",
              BuildKernelDefConstraintsFromTypeList<ValuesTypes>(),
              BuildKernelDefConstraintsFromTypeList<EnabledValuesTypes>()),
      OneHot);

// T1: indices, T2: depth, T3: values
ONNX_CPU_OPERATOR_KERNEL(
      OneHot,
      11,
      KernelDefBuilder()
          .TypeConstraint(
              "T1",
              BuildKernelDefConstraintsFromTypeList<IndicesTypes>(),
              BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>())
          .TypeConstraint(
              "T2",
              BuildKernelDefConstraintsFromTypeList<DepthTypes>(),
              BuildKernelDefConstraintsFromTypeList<EnabledDepthTypes>())
          .TypeConstraint(
              "T3",
              BuildKernelDefConstraintsFromTypeList<ValuesTypes>(),
              BuildKernelDefConstraintsFromTypeList<EnabledValuesTypes>()),
      OneHot);

Status ValidateInputs(const Tensor* depth, const Tensor* values) {
  // validation scenarios
  // depth should be scalar and > 0
  if (!depth->Shape().IsScalar()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid argument for depth; it's not a scalar.");
  }

  // values Rank 1 tensor containing exactly two elements
  if (!(values->Shape().NumDimensions() == 1 && values->Shape().Size() == 2)) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Invalid argument for values; either it's rank is more than 1"
                  " or it has more than 2 elements");
  }

  return Status::OK();
}

namespace {
template <typename Iterator>
int64_t prod(Iterator begin, Iterator end) {
  return std::accumulate(begin, end, 1, std::multiplies<int64_t>());
}
}  // namespace

Status PrepareOutputShape(const Tensor* indices, const int64_t depth_val, const int64_t axis,
                          int64_t& prefix_dim_size, int64_t& suffix_dim_size,
                          TensorShapeVector& output_shape) {
  const auto& indices_shape = indices->Shape();
  const auto indices_num_dims = indices_shape.NumDimensions();
  output_shape = indices_shape.AsShapeVector();

  // output rank is always 1 more than the input rank as a new dimension is added to the input shape
  const auto output_rank = static_cast<int64_t>(indices_num_dims) + 1;

  auto true_axis = HandleNegativeAxis(axis, output_rank);

  output_shape.insert(output_shape.begin() + true_axis, depth_val);

  prefix_dim_size = prod(output_shape.begin(), output_shape.begin() + true_axis);
  suffix_dim_size = prod(output_shape.begin() + true_axis + 1, output_shape.end());

  return Status::OK();
}

namespace {
template <typename dtype>
int64_t scalar_value(const Tensor* scalar_tensor) {
  const auto* data = scalar_tensor->Data<dtype>();
  return static_cast<int64_t>(*data);
}

template <typename depth_type>
struct DepthDispatchTarget {
  void operator()(const Tensor* depth, int64_t& depth_val) {
    depth_val = scalar_value<depth_type>(depth);
  }
};

// Type is a choice of memory footprint versus supported numerical range for depth.
// TODO: consider int16_t or uint16_t if no one uses depth > 2^15 or > 2^16
typedef int32_t AdjustedIndicesType;

// Depth must be in the numeric range of AdjustedIndicesType to enable
// adjusting all negative and out-of-range indices.
constexpr int64_t kMaxDepth = std::numeric_limits<AdjustedIndicesType>::max();

static_assert(double(kMaxDepth) == double(std::numeric_limits<AdjustedIndicesType>::max()),
              "sanity check that AdjustedIndicesType isn't u64 or floating point");

template <typename in_type>
AdjustedIndicesType adjust(in_type index, int64_t depth_val) {
  int64_t i = static_cast<int64_t>(index);
  if (i < 0) {
    i += depth_val;
  }
  if (i < std::numeric_limits<AdjustedIndicesType>::min() || i > std::numeric_limits<AdjustedIndicesType>::max()) {
    // When in_type is larger than AdjustedIndicesType avoid that the cast at
    // the end erroneously returns a value in range [0,depth_val).
    i = depth_val;
  }
  return static_cast<AdjustedIndicesType>(i);
}

template <typename in_type>
struct AdjustIndicesDispatchTarget {
  void operator()(const Tensor* indices, int64_t indices_size, std::vector<AdjustedIndicesType>& adjusted_indices,
                  int64_t depth_val) {
    const auto* indices_data = indices->Data<in_type>();
    for (int64_t i = 0; i < indices_size; ++i) {
      adjusted_indices.push_back(adjust(indices_data[i], depth_val));
    }
  }
};

// Helper to define Tensor types given that the scalar is of type T.
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct EigenTensorTypes {
  using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using ConstEigenTensorMap = Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using Scalar = Eigen::TensorMap<Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using ConstScalar = Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
  using ConstMatrix = Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned>;
};

namespace generator {
template <typename in_type, typename out_type>
class OneGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  OneGenerator(const typename EigenTensorTypes<in_type>::ConstMatrix& indices,
               const typename EigenTensorTypes<out_type>::ConstScalar& on_value,
               const typename EigenTensorTypes<out_type>::ConstScalar& off_value)
      : indices_(indices), on_value_(on_value), off_value_(off_value) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE out_type
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& pre_depth_suff) const {
    return (indices_(pre_depth_suff[0], pre_depth_suff[2]) == pre_depth_suff[1])
               ? on_value_()
               : off_value_();
  }

 private:
  const typename EigenTensorTypes<in_type>::ConstMatrix indices_;
  const typename EigenTensorTypes<out_type>::ConstScalar on_value_;
  const typename EigenTensorTypes<out_type>::ConstScalar off_value_;
};
}  // namespace generator

template <typename out_type>
struct GenerateOutputDispatchTarget {
  void operator()(const Tensor* values,
                  const typename EigenTensorTypes<AdjustedIndicesType>::ConstMatrix& indices_tensor_e,
                  Tensor* output,
                  Eigen::array<Eigen::DenseIndex, 3> output_dims_e) {
    auto* output_data = output->MutableData<out_type>();
    typename EigenTensorTypes<out_type, 3>::EigenTensorMap output_tensor_e(output_data, output_dims_e);

    const auto* values_data = values->Data<out_type>();
    typename EigenTensorTypes<out_type>::ConstScalar on_value_e(values_data + 1);
    typename EigenTensorTypes<out_type>::ConstScalar off_value_e(values_data);

    generator::OneGenerator<AdjustedIndicesType, out_type> generator(indices_tensor_e, on_value_e, off_value_e);

    // TODO potential optimization opportunity
    // TODO figure out the eigen threadpool stuff for use here
    output_tensor_e = output_tensor_e.generate(generator);
  }
};
}  // namespace

Status OneHot::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* indices = p_op_kernel_context->Input<Tensor>(0);
  const auto* depth = p_op_kernel_context->Input<Tensor>(1);
  const auto* values = p_op_kernel_context->Input<Tensor>(2);

  ORT_RETURN_IF_ERROR(ValidateInputs(depth, values));

  int64_t depth_val; // As per spec in case 'depth' is of non-integer type, it will be casted to int64 before use.

  utils::MLTypeCallDispatcherFromTypeList<EnabledDepthTypes> depth_dispatcher{depth->GetElementType()};
  depth_dispatcher.Invoke<DepthDispatchTarget>(depth, depth_val);

  if (depth_val <= 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Depth is negative.");
  }
  if (depth_val > kMaxDepth) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Exceeded implementation's max depth " + std::to_string(kMaxDepth));
  }

  // prepare output shape
  int64_t prefix_dim_size, suffix_dim_size;
  TensorShapeVector output_shape;
  ORT_RETURN_IF_ERROR(PrepareOutputShape(indices, depth_val, axis_, prefix_dim_size, suffix_dim_size, output_shape));

  // allocate output
  Tensor* output = p_op_kernel_context->Output(0, TensorShape(output_shape));

  // edge case where we have a dim with a value of 0
  if (output->Shape().Size() == 0)
    return Status::OK();

  // Handle negative indices. It's faster to create a new indices instead of comparing in generator
  // since generator has much larger loops.
  const auto indices_size = indices->Shape().Size();
  std::vector<AdjustedIndicesType> adjusted_indices;
  adjusted_indices.reserve(indices_size);

  utils::MLTypeCallDispatcherFromTypeList<EnabledIndicesTypes> indices_dispatcher{indices->GetElementType()};
  indices_dispatcher.Invoke<AdjustIndicesDispatchTarget>(indices, indices_size, adjusted_indices, depth_val);

  // Split indices into matrix of size prefix_dim_size x suffix_dim_size
  Eigen::array<Eigen::DenseIndex, 2> indices_dims_e = {
      {static_cast<Eigen::DenseIndex>(prefix_dim_size), static_cast<Eigen::DenseIndex>(suffix_dim_size)}};
  typename EigenTensorTypes<AdjustedIndicesType, 2>::ConstEigenTensorMap indices_tensor_e(adjusted_indices.data(), indices_dims_e);

  // Split output into 3-Tensor of size:
  //   prefix_dim_size x depth x suffix_dim_size.
  Eigen::array<Eigen::DenseIndex, 3> output_dims_e = {{
      static_cast<Eigen::DenseIndex>(prefix_dim_size),
      static_cast<Eigen::DenseIndex>(depth_val),
      static_cast<Eigen::DenseIndex>(suffix_dim_size)
  }};

  utils::MLTypeCallDispatcherFromTypeList<EnabledValuesTypes> values_dispatcher{values->GetElementType()};
  values_dispatcher.Invoke<GenerateOutputDispatchTarget>(values, indices_tensor_e, output, output_dims_e);

  return Status::OK();
}
}  // namespace onnxruntime
