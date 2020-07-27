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
  SingleBlockImplicitTerminator<"DoneOp">,
  Symbol,
  RegionKindInterface
]> {
  let summary = "pipeline operation";
  let description = [{
    The "staticlogic.pipeline" operation represents a statically scheduled 
    pipeline stucture which can contain multiple pipeline stages.
  }];

  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion : $body);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<"OpBuilder &builder, OperationState &odsState, "
                            "StringRef name, FunctionType type, "
                            "ValueRange operands, " 
                            "ArrayRef<NamedAttribute> attributes", [{
    odsState.addAttribute(SymbolTable::getSymbolAttrName(),
                          builder.getStringAttr(name));
    odsState.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    odsState.addOperands(operands);
    odsState.addAttributes(attributes);
    for (unsigned i = 0; i != 1; ++i)
      (void)odsState.addRegion();
    
    auto *bodyRegion = odsState.regions[0].get();
    PipelineOp::ensureTerminator(*bodyRegion, builder, odsState.location);
  }]>];

  let extraClassDeclaration = [{
    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<PipelineOp>;

    /// Returns the number of arguments, implementing OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }
    /// Returns the number of results, implementing OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    ///  safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];

  let parser = [{ }];
  let printer = [{
    auto type = getAttrOfType<TypeAttr>(getTypeAttrName()).getValue()
                .cast<FunctionType>();
    impl::printFunctionLikeOp(p, *this, type.getInputs(), 
                              /*isVariadic=*/true, type.getResults());
  }];
  let verifier = [{ }];
}

def DoneOp : Op<StaticLogic_Dialect, "done", [Terminator]> {
  let summary = "StaticLogic termination operation";
  let description = [{
    "staticlogic.done" marks the end of a region in the StaticLogic dialect.
  }];

  let arguments = (ins);
}

#endif // STATICLOGIC_OPS