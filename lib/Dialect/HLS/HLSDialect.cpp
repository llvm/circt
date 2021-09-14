//===- HLSDialect.cpp - Implement the HLS dialect -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HLS dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HLS/HLSDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace hls;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Attribute storage
//===----------------------------------------------------------------------===//

/// An HLS directive is parsed through use of its TableGen'erated
/// implementation parser.
template <typename TDirectiveImpl>
static Attribute parseDirective(DialectAsmParser &parser, Type type) {
  TDirectiveImpl directive;
  if (parser.parseLess() ||
      parser.parseAttribute<TDirectiveImpl>(directive, type) ||
      parser.parseGreater())
    return Attribute();

  return HLSDirective<TDirectiveImpl>::get(directive);
}

Attribute HLSDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  llvm::StringRef attrKeyword;
  // parse directive keyword first
  if (parser.parseKeyword(&attrKeyword))
    return Attribute();

#define tryParseDirective(TDirectiveImpl)                                      \
  if (attrKeyword == HLSDirective<TDirectiveImpl>::getKeyword())               \
    return parseDirective<TDirectiveImpl>(parser, type);

  tryParseDirective(PipelineDirective);
  tryParseDirective(FunctionInstantiateDirective);

  emitError(parser.getEncodedSourceLoc(parser.getCurrentLocation()),
            "Invalid HLS attribute!");
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Attribute printing
//===----------------------------------------------------------------------===//

/// An HLS directive is printed through use of its TableGen'erated
/// implementation.
void HLSDialect::printAttribute(Attribute attr,
                                DialectAsmPrinter &printer) const {
  TypeSwitch<Attribute>(attr)
      .Case<HLSDirective<PipelineDirective>,
            HLSDirective<FunctionInstantiateDirective>>([&](auto dir) {
        printer << decltype(dir)::getKeyword() << "<";
        // Print the content of the directive through the directive
        // implementation.
        dir.impl().print(printer.getStream());
      })
      .Default([&](auto) { llvm_unreachable("Unknown HLS directive!"); });

  printer << ">";
}

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void HLSDialect::initialize() {
  addAttributes<HLSDirective<PipelineDirective>>();
  addAttributes<HLSDirective<FunctionInstantiateDirective>>();
}

#include "circt/Dialect/HLS/HLSAttributes.cpp.inc"
#include "circt/Dialect/HLS/HLSDialect.cpp.inc"
#include "circt/Dialect/HLS/HLSEnums.cpp.inc"
