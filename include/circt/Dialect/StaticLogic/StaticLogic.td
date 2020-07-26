//===- StaticLogic.td - StaticLogic operation definitions --*- tablegen -*-===//
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

#ifdef STATICLOGIC_OPS
#else
#define STATICLOGIC_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def StaticLogic_Dialect : Dialect {
  let name = "staticlogic";
  let cppNamespace = "";
}

def PipelineOp : Op<StaticLogic_Dialect, "pipeline", [
  NativeOpTrait<"IsIsolatedFromAbove">,
  NativeOpTrait<"FunctionLike">,
  Symbol,
  RegionKindInterface
]> {
  let summary = "pipeline operation";
  let description = [{
    The "pipeline" operation represents a statically scheduled pipeline stucture
    which contains multiple pipeline stages.
  }];

  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
}

#endif // STATICLOGIC_OPS