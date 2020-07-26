//===- StaticLogic.h - StaticLogic MLIR Operations --------------*- C++ -*-===//
//
// Copyright 2020 The CIRCT Authors.
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
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STATICLOGIC_OPS_H_
#define CIRCT_STATICLOGIC_OPS_H_

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
namespace staticlogic {

using namespace mlir;

class StaticLogicDialect : public Dialect {
public:
  StaticLogicDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "staticlogic"; }
};

#define GET_OP_CLASSES
#include "circt/Dialect/StaticLogic/StaticLogic.h.inc"

} // namespace staticlogic
} // end namespace circt
#endif // CIRCT_STATICLOGIC_OPS_H_
