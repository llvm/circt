//===- CalyxOps.cpp - Calyx op code defs ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is where op definitions live.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::calyx;
using namespace mlir;

/// Prints the name and width for Calyx component input and output ports.
/// For example,
/// ```
/// p1: 32, p2: 64
/// ```
void printComponentPortNameAndWidth(OpAsmPrinter &p, DictionaryAttr &nameToWidth) {
  for (auto i = nameToWidth.begin(), e = nameToWidth.end(); i < e; ++i) {
    p << i->first << ": " << i->second;
    if (i + 1 != e)
      p << ", ";
  }
}

//===----------------------------------------------------------------------===//
// ComponentOp
//===----------------------------------------------------------------------===//

static void printComponentOp(OpAsmPrinter &p, ComponentOp &op) {
  p << "component (";

  auto inPortNameAndWidth = op->getAttrOfType<DictionaryAttr>("inPortToWidth");
  printComponentPortNameAndWidth(p, inPortNameAndWidth);

  p << ") -> (";

  auto outPortNameAndWidth = op->getAttrOfType<DictionaryAttr>("outPortToWidth");
  printComponentPortNameAndWidth(p, outPortNameAndWidth);

  // TODO(calyx): print groups and control.
  p << ") {}";
}

void ComponentOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, DictionaryAttr inPortToWidth,
                        DictionaryAttr outPortToWidth) {
  // Add an attribute for the name.
  result.addAttribute(builder.getIdentifier("name"), name);

  if (inPortToWidth)
    result.addAttribute("inPortToWidth", inPortToWidth);
  if (outPortToWidth)
    result.addAttribute("outPortToWidth", outPortToWidth);

  // TODO(calyx): Add scaffolding for cells, wire, & control.
}

static ParseResult parseComponentOp(OpAsmParser &parser,
                                    OperationState &result) {
  return failure();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Calyx/Calyx.cpp.inc"
