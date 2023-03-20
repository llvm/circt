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
include "circt/Dialect/Arc/Types.td"
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

  let hasRegionVerifier = 1;

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
  ParentOneOf<["DefineOp", "LutOp"]>,
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
  AttrSizedOperandSegments
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

  let hasCanonicalizeMethod = 1;

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

def ClockGateOp : ArcOp<"clock_gate", [Pure]> {
  let summary = "Clock gate";
  let arguments = (ins I1:$input, I1:$enable);
  let results = (outs I1:$output);
  let assemblyFormat = [{
    $input `,` $enable attr-dict
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$enable), [{
      build($_builder, $_state, input.getType(), input, enable);
    }]>
  ];
}

def MemoryOp : ArcOp<"memory", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Memory";
  let results = (outs MemoryType:$memory);
  let assemblyFormat = [{
    type($memory) attr-dict
  }];
}

class MemoryAndDataTypesMatch<string mem, string data> : TypesMatchWith<
  "memory and data types must match", mem, data,
  "$_self.cast<MemoryType>().getWordType()">;

def MemoryReadOp : ArcOp<"memory_read", [
  MemoryEffects<[MemRead]>,
  MemoryAndDataTypesMatch<"memory", "data">
]> {
  let summary = "Read word from memory";
  let arguments = (ins
    MemoryType:$memory,
    AnyInteger:$address,
    I1:$clock,
    I1:$enable
  );
  let results = (outs AnyInteger:$data);
  let assemblyFormat = [{
    $memory `[` $address `]` `,` $clock `,` $enable
    attr-dict `:` type($memory) `,` type($address)
  }];
}

def MemoryWriteOp : ArcOp<"memory_write", [
  MemoryEffects<[MemWrite]>,
  MemoryAndDataTypesMatch<"memory", "data">,
  AttrSizedOperandSegments
]> {
  let summary = "Write word to memory";
  let arguments = (ins
    MemoryType:$memory,
    AnyInteger:$address,
    I1:$clock,
    I1:$enable,
    AnyInteger:$data,
    Optional<AnyInteger>:$mask,
    Variadic<AnyType>:$reads
  );
  let assemblyFormat = [{
    $memory `[` $address `]` `,` $clock `,` $enable `,` $data
    ( `mask` `(` $mask^ `:` type($mask) `)`)?
    (` ` `(` `reads` $reads^ `:` type($reads) `)`)?
    attr-dict `:` type($memory) `,` type($address)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Miscellaneous
//===----------------------------------------------------------------------===//

def TapOp : ArcOp<"tap"> {
  let summary = "A tracker op to observe a value under a given name";
  let arguments = (ins AnySignlessInteger:$value, StrAttr:$name);
  let assemblyFormat = [{ $value attr-dict `:` type($value) }];
}

def LutOp : ArcOp<"lut", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"arc::OutputOp">,
  Pure
]> {
  let summary = "A lookup-table.";
  let description = [{
    Represents a lookup-table as one operation. The operations that map the
    lookup/input values to the corresponding table-entry are collected inside
    the body of this operation.
    Note that the operation is marked to be isolated from above to guarantee
    that all input values have to be passed as an operand. This allows for
    simpler analyses and canonicalizations of the LUT as well as lowering.
    Only combinational operations are allowed inside the LUT, i.e., no
    side-effects, state, time delays, etc.
  }];

  let arguments = (ins Variadic<AnySignlessInteger>:$inputs);
  let results = (outs AnySignlessInteger:$output);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` $inputs `)` `:` functional-type($inputs, $output)
    attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    mlir::Block *getBodyBlock() { return &getBody().front(); }
  }];

  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_ARC_OPS_TD
