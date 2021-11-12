//===- FIRRTLOpInterfaces.cpp - Implement the FIRRTL op interfaces --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the FIRRTL operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace llvm;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// FlowKind
//===----------------------------------------------------------------------===//

Flow circt::firrtl::flipFlow(Flow flow) {
  switch (flow) {
  case Flow::Source:
    return Flow::Sink;
  case Flow::Sink:
    return Flow::Source;
  case Flow::Duplex:
    return Flow::Duplex;
  }
}

static Flow mergeFlow(Flow a, Flow b) {
  switch (a) {
  case Flow::Source:
    return b;
  case Flow::Sink:
    return flipFlow(b);
  case Flow::Duplex:
    return Flow::Duplex;
  }
}

Flow FlowKind::get(Value value) {
  Flow flow = Flow::Source;

  do {
    if (auto blockArg = value.dyn_cast<BlockArgument>()) {
      auto op = dyn_cast<FlowKind>(value.getParentBlock()->getParentOp());
      // If the operation doesn't implement the interface, return the result
      // so far.
      if (!op)
        return flow;
      auto result = op.getBlockArgFlow(blockArg);
      flow = mergeFlow(result.getFlow(), flow);
      value = result.getValue();
    } else {
      auto op = dyn_cast<FlowKind>(value.getDefiningOp());
      // If the operation doesn't implement the interface, return the result
      // so far.
      if (!op)
        return flow;
      auto result = op.getFlow(value.cast<OpResult>());
      flow = mergeFlow(result.getFlow(), flow);
      value = result.getValue();
    }
  } while (value && flow != Flow::Duplex);
  return flow;
}

//===----------------------------------------------------------------------===//
// FModuleLike
//===----------------------------------------------------------------------===//

LogicalResult circt::firrtl::verifyModuleLikeOpInterface(FModuleLike module) {
  // Verify port types first.  This is used as the basis for the number of
  // ports required everywhere else.
  auto portTypes = module.getPortTypesAttr();
  if (!portTypes)
    return module.emitOpError("requires valid port types");
  if (llvm::any_of(portTypes.getValue(), [](Attribute attr) {
        auto typeAttr = attr.dyn_cast<TypeAttr>();
        return !typeAttr || !typeAttr.getValue().isa<FIRRTLType>();
      }))
    return module.emitOpError("ports should all be FIRRTL types");

  auto numPorts = portTypes.size();

  // Verify the port dirctions.
  auto portDirections = module.getPortDirectionsAttr();
  if (!portDirections)
    return module.emitOpError("requires valid port direction");
  // TODO: bitwidth is 1 when there are no ports, since APInt previously did not
  // support 0 bit widths.
  auto bitWidth = portDirections.getValue().getBitWidth();
  if (bitWidth != numPorts)
    return module.emitOpError("requires ") << numPorts << " port directions";

  // Verify the port names.
  auto portNames = module.getPortNamesAttr();
  if (!portNames)
    return module.emitOpError("requires valid port names");
  if (portNames.size() != numPorts)
    return module.emitOpError("requires ") << numPorts << " port names";
  if (llvm::any_of(portNames.getValue(),
                   [](Attribute attr) { return !attr.isa<StringAttr>(); }))
    return module.emitOpError("port names should all be string attributes");

  // Verify the port annotations.
  auto portAnnotations = module.getPortAnnotationsAttr();
  if (!portNames)
    return module.emitOpError("requires valid port annotations");
  // TODO: it seems weird to allow empty port annotations.
  if (!portAnnotations.empty() && portAnnotations.size() != numPorts)
    return module.emitOpError("requires ") << numPorts << " port annotations";
  // TODO: Move this into an annotation verifier.
  for (auto annos : portAnnotations.getValue()) {
    auto arrayAttr = annos.dyn_cast<ArrayAttr>();
    if (!arrayAttr)
      return module.emitOpError(
          "requires port annotations be array attributes");
    if (llvm::any_of(arrayAttr.getValue(), [](Attribute attr) {
          return !attr.isa<DictionaryAttr, SubAnnotationAttr>();
        }))
      return module.emitOpError(
          "annotations must be dictionaries or subannotations");
  }

  // Verify the body.
  if (module->getNumRegions() != 1)
    return module.emitOpError("requires one region");

  // Verify the block arguments.
  auto &body = module->getRegion(0);
  if (!body.empty()) {
    auto &block = body.front();
    if (block.getNumArguments() != numPorts)
      return module.emitOpError("entry block must have ")
             << numPorts << " arguments to match module signature";

    if (llvm::any_of(
            llvm::zip(block.getArguments(), portTypes.getValue()),
            [](auto pair) {
              auto blockType = std::get<0>(pair).getType();
              auto portType =
                  std::get<1>(pair).template cast<TypeAttr>().getValue();
              return blockType != portType;
            }))
      return module.emitOpError(
          "block argument types should match signature types");
  }

  return success();
}

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp.inc"
