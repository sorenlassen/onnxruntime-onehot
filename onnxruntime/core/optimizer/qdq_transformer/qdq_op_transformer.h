// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

constexpr char* QOPTypeName = "QuantizeLinear";
constexpr char* DQOPTypeName = "DequantizeLinear";

class Node;
class Graph;

class QDQOperatorTransformer {
 public:
  QDQOperatorTransformer(Node& node, Graph& graph) : node_(node), graph_(graph) {}
  virtual ~QDQOperatorTransformer() {}
  virtual bool Transform(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) = 0;

  /* Determine whether to keep node_ itself or not.
           For operators that support int8, keep node_ and only change its input and output.
           Otherwise, node_ will be removed and replaced by a QLinear* version.
        */
  virtual bool KeepNode() const {
    return false;
  }

  void FillQDQOptionalZeroPoint(const std::vector<const Node*>& parents);

 protected:
  Node& node_;
  Graph& graph_;

  static const ONNX_NAMESPACE::TensorProto optional_zero_point_int8_;
  static const ONNX_NAMESPACE::TensorProto optional_zero_point_uint8_;
};
}  // namespace onnxruntime
