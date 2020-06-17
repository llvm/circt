//===- StandardToHandshake.h ------------------------------------*- C++ -*-===//
//
// Copyright 2019 The CIRCT Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef MLIR_CONVERSION_STANDARDTOHANDSHAKE_H_
#define MLIR_CONVERSION_STANDARDTOHANDSHAKE_H_

#include <memory>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

namespace mlir {
  namespace handshake {
    class FuncOp;
    void registerStandardToHandshakePasses();
  }
} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOHANDSHAKE_H_
