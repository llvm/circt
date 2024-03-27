//===- OMTypes.cpp - Object Model type definitions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model type definitions.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMTypes.h"
#include "circt/Dialect/OM/OMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace mlir;
using namespace circt::om;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/OM/OMTypes.cpp.inc"

void circt::om::OMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/OM/OMTypes.cpp.inc"
      >();
}

mlir::LogicalResult
circt::om::MapType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> diag,
                           mlir::Type keyType, mlir::Type elementType) {
  if (!llvm::isa<om::StringType, mlir::IntegerType>(keyType))
    return diag() << "map key type must be either string or integer but got "
                  << keyType;
  return mlir::success();
}

bool circt::om::isMapKeyValuePairType(mlir::Type type) {
  auto tuple = llvm::dyn_cast<mlir::TupleType>(type);
  return tuple && tuple.getTypes().size() == 2 &&
         llvm::isa<om::StringType, mlir::IntegerType>(tuple.getTypes().front());
}
