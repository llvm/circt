//===- RTGTestTypes.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTGTest/IR/RTGTestTypes.h"
#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyAttrInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyTypeInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "circt/Dialect/RTGTest/IR/RTGTestAttributes.h"
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace rtgtest;

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTGTest/IR/RTGTestTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// IntegerRegisterType
//===----------------------------------------------------------------------===//

bool IntegerRegisterType::isValidContentType(Type type) const {
  return llvm::isa<rtg::ImmediateType>(type);
}

TypedAttr IntegerRegisterType::parseContentValue(llvm::StringRef valueString,
                                                 Type contentType) const {
  auto immType = dyn_cast<rtg::ImmediateType>(contentType);
  if (!immType)
    return {};

  APInt intValue;
  if (valueString.getAsInteger(0, intValue))
    return {};

  if (intValue.getBitWidth() < immType.getWidth()) {
    intValue = intValue.zext(immType.getWidth());
  } else {
    if (!intValue.isIntN(immType.getWidth()))
      return {};
    intValue = intValue.trunc(immType.getWidth());
  }

  return rtg::ImmediateAttr::get(getContext(), intValue);
}

std::string IntegerRegisterType::getIntrinsicLabel(Attribute value,
                                                   llvm::StringRef id) const {
  return "spike.pre.printreg.x" +
         std::to_string(
             cast<rtg::RegisterAttrInterface>(value).getClassIndex()) +
         "." + id.str();
}

unsigned IntegerRegisterType::getRegisterClassSize() { return 32; }

rtg::RegisterAttrInterface
IntegerRegisterType::getRegisterAttrForClassIndex(MLIRContext *ctx,
                                                  unsigned index) {
  switch (index) {
  case 0:
    return RegZeroAttr::get(ctx);
  case 1:
    return RegRaAttr::get(ctx);
  case 2:
    return RegSpAttr::get(ctx);
  case 3:
    return RegGpAttr::get(ctx);
  case 4:
    return RegTpAttr::get(ctx);
  case 5:
    return RegT0Attr::get(ctx);
  case 6:
    return RegT1Attr::get(ctx);
  case 7:
    return RegT2Attr::get(ctx);
  case 8:
    return RegS0Attr::get(ctx);
  case 9:
    return RegS1Attr::get(ctx);
  case 10:
    return RegA0Attr::get(ctx);
  case 11:
    return RegA1Attr::get(ctx);
  case 12:
    return RegA2Attr::get(ctx);
  case 13:
    return RegA3Attr::get(ctx);
  case 14:
    return RegA4Attr::get(ctx);
  case 15:
    return RegA5Attr::get(ctx);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// FloatRegisterType
//===----------------------------------------------------------------------===//

unsigned FloatRegisterType::getRegisterClassSize() { return 1; }

rtg::RegisterAttrInterface
FloatRegisterType::getRegisterAttrForClassIndex(MLIRContext *ctx,
                                                unsigned index) {
  switch (index) {
  case 0:
    return RegF0Attr::get(ctx);
  default:
    return {};
  }
}

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

void circt::rtgtest::RTGTestDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/RTGTest/IR/RTGTestTypes.cpp.inc"
      >();
}
