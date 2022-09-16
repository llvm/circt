//===- SystemCExpressions.td - SystemC expression ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemC expressions.
//
//===----------------------------------------------------------------------===//

def SignalReadOp : SystemCOp<"signal.read", [NoSideEffect,
                                             InferTypeOpInterface]> {
  let summary = "Returns the current value of a signal or port.";
  let description = [{
    Represents the member function `const T& read() const;` and operator
    `operator const T& () const;` of class `sc_signal` as described in
    IEEE 1666-2011 §6.4.7., of class `sc_in` as described in §6.8.3., and
    of class `sc_inout` as decribed in §6.10.3. It shall return the current
    value of the signal/port.
  }];

  let arguments = (ins AnyTypeOf<[InputType, InOutType, SignalType]>:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input))";

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          Optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results) {
      results.push_back(systemc::getSignalBaseType(operands[0].getType()));
      return success();
    }
  }];
}

def ConvertOp : SystemCOp<"convert", [NoSideEffect]> {
  let summary = "Converts between various integer and bit vector types.";
  let description = [{
    Allows conversions between the various integer and bit vector types in
    SystemC, including MLIRs signless integers that are used to represent the
    primitive C integer types, according to the explicit and implicit
    constructors, implicit access operators and explicit conversion member
    functions defined in the respective data type class in the SystemC spec
    (refer to the description of the supported types and the spec chapters
    listed there for more information).
  }];

  let arguments = (ins AnyTypeOf<[AnySystemCInteger,
                                  AnySystemCVector,
                                  LogicType]>:$input);

  let results = (outs AnyTypeOf<[AnySystemCInteger,
                                 AnySystemCVector,
                                 LogicType]>:$result);

  let assemblyFormat = "$input attr-dict `:` functional-type($input, $result)";
}

//===----------------------------------------------------------------------===//
// Operations that model C++-level features.
//===----------------------------------------------------------------------===//

def NewOp : SystemCOp<"cpp.new", [NoSideEffect]> {
  let summary = "A C++ 'new' expression.";
  let description = [{
    Creates and initializes C++ objects using dynamic storage.
    Note that the types of the constructor arguments are not verified in any way
    w.r.t. the result's type because it is allowed to use an opaque type for the
    pointee.
  }];

  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs EmitC_PointerType:$result);

  let assemblyFormat = [{
    `(` $args `)` attr-dict `:` functional-type($args, $result)
  }];
}
