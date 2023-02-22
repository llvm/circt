//===- Ops.td - Arc dialect operations ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_OPS_TD
#define CIRCT_DIALECT_ARC_OPS_TD

include "circt/Dialect/Arc/Dialect.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class ArcOp<string mnemonic, list<Trait> traits = []> :
  Op<ArcDialect, mnemonic, traits>;

def DefineOp : ArcOp<"define", [
  IsolatedFromAbove,
  FunctionOpInterface,
  Symbol,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"arc::OutputOp">,
  HasParent<"mlir::ModuleOp">
]> {
  let summary = "State transfer arc definition";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$sym_name, "mlir::TypeAttr":$function_type), [{
      build($_builder, $_state, sym_name, function_type, mlir::ArrayAttr(), mlir::ArrayAttr());
    }]>,
    OpBuilder<(ins "mlir::StringRef":$sym_name, "mlir::FunctionType":$function_type), [{
      build($_builder, $_state, sym_name, function_type, mlir::ArrayAttr(), mlir::ArrayAttr());
    }]>,
  ];

  let extraClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::SSACFG;
    }

    mlir::Block &getBodyBlock() { return getBody().front(); }

    // Get the arc's symbolic name.
    mlir::StringAttr getNameAttr() {
      return (*this)->getAttrOfType<mlir::StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the arc's symbolic name.
    mlir::StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    mlir::LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<mlir::FunctionType>())
        return emitOpError("requires '") << getFunctionTypeAttrName() <<
                           "' attribute of function type";
      return mlir::success();
    }
  }];
}

def OutputOp : ArcOp<"output", [
  Terminator,
  ParentOneOf<["DefineOp"]>,
  Pure,
  ReturnLike
]> {
  let summary = "Arc terminator";
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` qualified(type($outputs)))?
  }];
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
  let hasVerifier = 1;
}

def StateOp : ArcOp<"state", [
  CallOpInterface, MemRefsNormalizable,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  AttrSizedOperandSegments, Pure
]> {
  let summary = "State transfer arc";

  let arguments = (ins
    FlatSymbolRefAttr:$arc,
    Optional<I1>:$clock,
    Optional<I1>:$enable,
    I32Attr:$latency,
    Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $arc `(` $inputs `)` (`clock` $clock^)? (`enable` $enable^)? `lat` $latency attr-dict
    `:` functional-type($inputs, results)
  }];

  let builders = [
    OpBuilder<(ins "DefineOp":$arc, "mlir::Value":$clock, "mlir::Value":$enable,
      "unsigned":$latency, CArg<"mlir::ValueRange", "{}">:$inputs), [{
      build($_builder, $_state, mlir::SymbolRefAttr::get(arc),
            arc.getFunctionType().getResults(), clock, enable, latency,
            inputs);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$arc, "mlir::TypeRange":$results, "mlir::Value":$clock,
      "mlir::Value":$enable, "unsigned":$latency, CArg<"mlir::ValueRange", "{}">:$inputs
    ), [{
      if (clock)
        $_state.addOperands(clock);
      if (enable)
        $_state.addOperands(enable);
      $_state.addOperands(inputs);
      $_state.addAttribute("arc", arc);
      $_state.addAttribute("latency", $_builder.getI32IntegerAttr(latency));
      $_state.addAttribute(getOperandSegmentSizeAttr(),
        $_builder.getDenseI32ArrayAttr({
          clock ? 1 : 0,
          enable ? 1 : 0,
          static_cast<int32_t>(inputs.size())}));
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "mlir::StringAttr":$arc, "mlir::TypeRange":$results, "mlir::Value":$clock,
      "mlir::Value":$enable, "unsigned":$latency, CArg<"mlir::ValueRange", "{}">:$inputs
    ), [{
      build($_builder, $_state, mlir::SymbolRefAttr::get(arc), results, clock, enable,
            latency, inputs);
    }]>,
    OpBuilder<(ins "mlir::StringRef":$arc, "mlir::TypeRange":$results, "mlir::Value":$clock,
      "mlir::Value":$enable, "unsigned":$latency, CArg<"mlir::ValueRange", "{}">:$inputs
    ), [{
      build($_builder, $_state, mlir::StringAttr::get($_builder.getContext(), arc),
            results, clock, enable, latency, inputs);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return {operand_begin(), operand_end()};
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("arc");
    }
  }];
}

#endif // CIRCT_DIALECT_ARC_OPS_TD
