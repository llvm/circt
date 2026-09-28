//===- DPITypeInfo.cpp - DPI type mapping
//----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/DPITypeInfo.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace circt;
using namespace mlir;
using sv::DPIIntegerContext;
using sv::DPITypeInfo;

StringRef DPITypeInfo::getIntegerKeyword() const {
  switch (kind) {
  case Kind::Bit:
    return "bit";
  case Kind::Logic:
    return "logic";
  case Kind::Byte:
    return "byte";
  case Kind::ShortInt:
    return "shortint";
  case Kind::Int:
    return "int";
  case Kind::LongInt:
    return "longint";
  default:
    return {};
  }
}

bool DPITypeInfo::isIntegerAtom() const {
  return kind == Kind::Byte || kind == Kind::ShortInt || kind == Kind::Int ||
         kind == Kind::LongInt;
}

bool DPITypeInfo::isValidReturn() const {
  return isIntegerAtom() ||
         ((kind == Kind::Bit || kind == Kind::Logic) && width == 1);
}

DPITypeInfo sv::getDPIIntegerTypeInfo(unsigned width,
                                      DPIIntegerContext context) {
  DPITypeInfo result{context == DPIIntegerContext::Typedef
                         ? DPITypeInfo::Kind::Logic
                         : DPITypeInfo::Kind::Bit};
  result.width = width;
  if (context != DPIIntegerContext::Import)
    return result;
  switch (width) {
  case 8:
    result.kind = DPITypeInfo::Kind::Byte;
    break;
  case 16:
    result.kind = DPITypeInfo::Kind::ShortInt;
    break;
  case 32:
    result.kind = DPITypeInfo::Kind::Int;
    break;
  case 64:
    result.kind = DPITypeInfo::Kind::LongInt;
    break;
  }
  result.isSigned = result.isIntegerAtom();
  return result;
}

DPITypeInfo sv::getDPIEnumTypeInfo(unsigned width) {
  DPITypeInfo result{DPITypeInfo::Kind::Enum};
  result.width = width;
  result.isSigned = width == 32;
  return result;
}

FailureOr<DPITypeInfo> sv::resolveDPIType(Type type) {
  auto integer = dyn_cast<IntegerType>(type);
  if (!integer || !integer.getWidth())
    return failure();
  return getDPIIntegerTypeInfo(integer.getWidth(), DPIIntegerContext::Import);
}
