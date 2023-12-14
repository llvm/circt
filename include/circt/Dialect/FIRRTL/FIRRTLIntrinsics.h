//===- FIRRTLDialect.h - FIRRTL dialect declaration ------------*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers for the lowering of intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_H

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

namespace circt {
namespace firrtl {
class InstanceGraph;

/// Base class for Intrinsic Converters.
///
/// Intrinsic converters contain validation logic, along with a converter
/// method to transform instances to extmodules/intmodules into ops.
class IntrinsicConverter {
protected:
  StringRef name;
  FModuleLike mod;

public:
  IntrinsicConverter(StringRef name, FModuleLike mod) : name(name), mod(mod) {}

  virtual ~IntrinsicConverter();

  /// Checks whether the intrinsic module is well-formed.
  ///
  /// This or's multiple ParseResults together, returning true on failure.
  virtual bool check() { return false; }

  /// Transform an instance of the intrinsic.
  virtual void convert(InstanceOp op) = 0;

protected:
  ParseResult hasNPorts(unsigned n);

  ParseResult namedPort(unsigned n, StringRef portName);

  ParseResult resetPort(unsigned n);

  ParseResult hasNParam(unsigned n, unsigned c = 0);

  ParseResult namedParam(StringRef paramName, bool optional = false);

  ParseResult namedIntParam(StringRef paramName, bool optional = false);

  template <typename T>
  ParseResult typedPort(unsigned n) {
    auto ports = mod.getPorts();
    if (n >= ports.size()) {
      mod.emitError(name) << " missing port " << n;
      return failure();
    }
    if (!isa<T>(ports[n].type)) {
      mod.emitError(name) << " port " << n << " not of correct type";
      return failure();
    }
    return success();
  }

  template <typename T>
  ParseResult sizedPort(unsigned n, int32_t size) {
    auto ports = mod.getPorts();
    if (failed(typedPort<T>(n)))
      return failure();
    if (cast<T>(ports[n].type).getWidth() != size) {
      mod.emitError(name) << " port " << n << " not size " << size;
      return failure();
    }
    return success();
  }
};

/// Lowering helper which collects all intrinsic converters.
class IntrinsicLowerings {
private:
  using ConverterFn = std::function<LogicalResult(FModuleLike)>;

  /// Reference to the MLIR context.
  MLIRContext *context;

  /// Reference to the instance graph to find module instances.
  InstanceGraph &graph;

  /// Mapping from intrinsic names to converters.
  DenseMap<StringAttr, ConverterFn> intmods;
  /// Mapping from extmodule names to converters.
  DenseMap<StringAttr, ConverterFn> extmods;

public:
  IntrinsicLowerings(MLIRContext *context, InstanceGraph &graph)
      : context(context), graph(graph) {}

  /// Registers a converter to an intrinsic name.
  template <typename T>
  void add(StringRef name) {
    addConverter<T>(intmods, name);
  }

  /// Registers a converter to an extmodule name.
  template <typename T>
  void addExtmod(StringRef name) {
    addConverter<T>(extmods, name);
  }

  /// Registers a converter to multiple intrinsic names.
  template <typename T, typename... Args>
  void add(StringRef name, Args... args) {
    add<T>(name);
    add<T>(args...);
  }

  /// Lowers a module to an intrinsic, given an intrinsic name.
  LogicalResult lower(CircuitOp circuit);

  /// Return the number of intrinsics converted.
  unsigned getNumConverted() const { return numConverted; }

private:
  template <typename T>
  void addConverter(DenseMap<StringAttr, ConverterFn> &map, StringRef name) {
    auto nameAttr = StringAttr::get(context, name);
    map.try_emplace(nameAttr, [&](FModuleLike mod) -> LogicalResult {
      T conv(name, mod);
      return doLowering(mod, conv);
    });
  }

  LogicalResult doLowering(FModuleLike mod, IntrinsicConverter &conv);

  unsigned numConverted = 0;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_H
