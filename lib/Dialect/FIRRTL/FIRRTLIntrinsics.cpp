//===- FIRRTLIntrinsics.cpp - Lower Intrinsics ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"

using namespace circt;
using namespace firrtl;

IntrinsicConverter::~IntrinsicConverter() = default;

ParseResult IntrinsicConverter::hasNPorts(unsigned n) {
  if (mod.getPorts().size() != n) {
    mod.emitError(name) << " has " << mod.getPorts().size()
                        << " ports instead of " << n;
    return failure();
  }
  return success();
}

ParseResult IntrinsicConverter::namedPort(unsigned n, StringRef portName) {
  auto ports = mod.getPorts();
  if (n >= ports.size()) {
    mod.emitError(name) << " missing port " << n;
    return failure();
  }
  if (!ports[n].getName().equals(portName)) {
    mod.emitError(name) << " port " << n << " named '" << ports[n].getName()
                        << "' instead of '" << portName << "'";
    return failure();
  }
  return success();
}

ParseResult IntrinsicConverter::resetPort(unsigned n) {
  auto ports = mod.getPorts();
  if (n >= ports.size()) {
    mod.emitError(name) << " missing port " << n;
    return failure();
  }
  if (isa<ResetType, AsyncResetType>(ports[n].type))
    return success();
  if (auto uintType = dyn_cast<UIntType>(ports[n].type))
    if (uintType.getWidth() == 1)
      return success();
  mod.emitError(name) << " port " << n << " not of correct type";
  return failure();
}

ParseResult IntrinsicConverter::hasNParam(unsigned n, unsigned c) {
  unsigned num = 0;
  if (mod.getParameters())
    num = mod.getParameters().size();
  if (num < n || num > n + c) {
    auto d = mod.emitError(name) << " has " << num << " parameters instead of ";
    if (c == 0)
      d << n;
    else
      d << " between " << n << " and " << (n + c);
    return failure();
  }
  return success();
}

ParseResult IntrinsicConverter::namedParam(StringRef paramName, bool optional) {
  for (auto a : mod.getParameters()) {
    auto param = cast<ParamDeclAttr>(a);
    if (param.getName().getValue().equals(paramName)) {
      if (isa<StringAttr>(param.getValue()))
        return success();

      mod.emitError(name) << " has parameter '" << param.getName()
                          << "' which should be a string but is not";
      return failure();
    }
  }
  if (optional)
    return success();
  mod.emitError(name) << " is missing parameter " << paramName;
  return failure();
}

ParseResult IntrinsicConverter::namedIntParam(StringRef paramName,
                                              bool optional) {
  for (auto a : mod.getParameters()) {
    auto param = cast<ParamDeclAttr>(a);
    if (param.getName().getValue().equals(paramName)) {
      if (isa<IntegerAttr>(param.getValue()))
        return success();

      mod.emitError(name) << " has parameter '" << param.getName()
                          << "' which should be an integer but is not";
      return failure();
    }
  }
  if (optional)
    return success();
  mod.emitError(name) << " is missing parameter " << paramName;
  return failure();
}

LogicalResult IntrinsicLowerings::doLowering(FModuleLike mod,
                                             IntrinsicConverter &conv) {
  if (conv.check())
    return failure();
  for (auto *use : graph.lookup(mod)->uses())
    conv.convert(use->getInstance<InstanceOp>());
  return success();
}

LogicalResult IntrinsicLowerings::lower(CircuitOp circuit) {
  unsigned numFailures = 0;
  for (auto op : llvm::make_early_inc_range(circuit.getOps<FModuleLike>())) {
    StringAttr intname;
    if (auto extMod = dyn_cast<FExtModuleOp>(*op)) {
      // Special-case some extmodules, identifying them by name.
      auto it = extmods.find(extMod.getDefnameAttr());
      if (it != extmods.end()) {
        if (succeeded(it->second(op))) {
          op.erase();
        } else {
          ++numFailures;
        }
        continue;
      }

      // Otherwise, find extmodules which have an intrinsic annotation.
      auto anno = AnnotationSet(&*op).getAnnotation("circt.Intrinsic");
      if (!anno)
        continue;
      intname = anno.getMember<StringAttr>("intrinsic");
      if (!intname) {
        op.emitError("intrinsic annotation with no intrinsic name");
        ++numFailures;
        continue;
      }
    } else if (auto intMod = dyn_cast<FIntModuleOp>(*op)) {
      intname = intMod.getIntrinsicAttr();
      if (!intname) {
        op.emitError("intrinsic module with no intrinsic name");
        ++numFailures;
        continue;
      }
    } else {
      continue;
    }

    // Find the converter and apply it.
    auto it = intmods.find(intname);
    if (it == intmods.end())
      return op.emitError() << "intrinsic not recognized";
    if (failed(it->second(op))) {
      ++numFailures;
      continue;
    }
    op.erase();
  }

  return success(numFailures == 0);
}
