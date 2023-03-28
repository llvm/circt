//===- ESITypes.cpp - ESI types code defs -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for esi data types. Anything which doesn't have to be public
// should go in here.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::esi;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.cpp.inc"

AnyType AnyType::get(MLIRContext *context) { return Base::get(context); }

LogicalResult
WindowType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                   StringAttr name, Type into,
                   ArrayRef<WindowFrameType> frames) {
  auto structInto = into.dyn_cast<hw::StructType>();
  if (!structInto)
    return emitError() << "only windows into structs are currently supported";

  auto fields = structInto.getElements();
  for (auto frame : frames) {
    DenseMap<StringAttr, WindowFieldType> frameFields;
    for (auto field : frame.getMembers())
      frameFields[field.getFieldName()] = field;

    while (!fields.empty() && !frameFields.empty()) {
      hw::StructType::FieldInfo field = fields.front();
      auto f = frameFields.find(field.name);
      if (f != frameFields.end()) {
        uint64_t numItems = f->getSecond().getNumItems();
        if (numItems > 0) {
          auto arrField = field.type.dyn_cast<hw::ArrayType>();
          if (!arrField)
            return emitError() << "cannot specify num items on non-array field "
                               << field.name;
          if (numItems > arrField.getSize())
            return emitError()
                   << "num items is larger than array size in field "
                   << field.name;
        }
        frameFields.erase(f);
      }
      fields = fields.drop_front();
    }
    if (!frameFields.empty())
      return emitError() << "invalid field name: "
                         << frameFields.begin()->getSecond().getFieldName();
  }
  return success();
}

void ESIDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/ESI/ESITypes.cpp.inc"
      >();
}
