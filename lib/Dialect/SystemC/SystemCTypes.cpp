//===- SystemCTypes.cpp - Implement the SystemC types ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemC dialect type system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCTypes.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::systemc;

Type systemc::getSignalBaseType(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<InputType, OutputType, InOutType, SignalType>(
          [](auto ty) { return ty.getBaseType(); })
      .Default([](auto ty) { return Type(); });
}

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SystemC/SystemCTypes.cpp.inc"

void SystemCDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/SystemC/SystemCTypes.cpp.inc"
      >();
}
