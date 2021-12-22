//===- FIRAnnotations.h - .fir file annotation interface --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains private APIs the parser uses to load annotations.
//
//===----------------------------------------------------------------------===//

#ifndef FIRANNOTATIONS_H
#define FIRANNOTATIONS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
namespace json {
class Path;
class Value;
} // namespace json
} // namespace llvm

namespace mlir {
class ArrayAttr;
}

namespace circt {
namespace firrtl {

class CircuitOp;
class PrintFOp;

bool fromJSON(llvm::json::Value &value, StringRef circuitTarget,
              llvm::StringMap<ArrayAttr> &annotationMap, llvm::json::Path path,
              CircuitOp circuit, size_t &nlaNumber);

/// Convert a JSON value containing OMIR JSON (an array of OMNodes), convert
/// this to an OMIRAnnotation, and add it to a mutable `annotationMap` argument.
bool fromOMIRJSON(llvm::json::Value &value, StringRef circuitTarget,
                  llvm::StringMap<ArrayAttr> &annotationMap,
                  llvm::json::Path path, MLIRContext *context);

bool scatterCustomMultiCircuitAnnotations(
    SmallVectorImpl<Attribute> &attrs,
    llvm::StringMap<ArrayAttr> &annotationMap,
    llvm::StringMap<ArrayAttr> &subcircuitAnnotationMap, CircuitOp circuit,
    unsigned &annotationID, Location loc, size_t &nlaNumber);

bool scatterCustomAnnotations(llvm::StringMap<ArrayAttr> &annotationMap,
                              CircuitOp circuit, unsigned &annotationID,
                              Location loc, size_t &nlaNumber);

bool fromJSONRaw(llvm::json::Value &value, StringRef circuitTarget,
                 SmallVectorImpl<Attribute> &attrs, llvm::json::Path path,
                 MLIRContext *context);

ParseResult foldWhenEncodedVerifOp(PrintFOp printOp);

} // namespace firrtl
} // namespace circt

#endif // FIRANNOTATIONS_H
