//===- Ops.h - Handshake MLIR Operations -----------------------------*- C++
//-*-===//
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
//
// This file defines convenience types for working with handshake operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_HANDSHAKEOPS_OPS_H_
#define CIRCT_HANDSHAKEOPS_OPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace handshake {

using namespace mlir;
class TerminatorOp;

class HandshakeOpsDialect : public Dialect {
public:
  HandshakeOpsDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "handshake"; }
};

#include "circt/Dialect/Handshake/HandshakeInterfaces.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Handshake/HandshakeOps.h.inc"

// template <typename T> struct FunctionPass : public OperationPass<T, FuncOp> {
//   /// The polymorphic API that runs the pass over the currently held
//   function. virtual void runOnFunction() = 0;

//   /// The polymorphic API that runs the pass over the currently held
//   operation. void runOnOperation() final {
//     if (!getFunction().isExternal())
//       runOnFunction();
//   }

//   /// Return the current module being transformed.
//   FuncOp getFunction() { return this->getOperation(); }
// };

} // end namespace handshake
} // end namespace circt
#endif // MLIR_HANDSHAKEOPS_OPS_H_
