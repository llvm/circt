//===- ESIAttributes.cpp - Implement ESI dialect attributes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Base64.h"

using namespace circt;
using namespace esi;

AppIDPathAttr AppIDPathAttr::getParent() {
  ArrayRef<AppIDAttr> path = getPath();
  if (path.empty())
    return {};
  return AppIDPathAttr::get(getContext(), getRoot(), path.drop_back());
}

Attribute BlobAttr::parse(AsmParser &odsParser, Type odsType) {
  std::string base64;
  if (odsParser.parseLess() || odsParser.parseString(&base64) ||
      odsParser.parseGreater())
    return {};
  std::vector<char> data;
  if (auto err = llvm::decodeBase64(base64, data)) {
    llvm::handleAllErrors(std::move(err), [&](const llvm::ErrorInfoBase &eib) {
      odsParser.emitError(odsParser.getNameLoc(), eib.message());
    });
    return {};
  }
  return BlobAttr::get(odsParser.getBuilder().getContext(), data);
}
void BlobAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<\"" << llvm::encodeBase64(getData()) << "\">";
}

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/ESI/ESIAttributes.cpp.inc"

void ESIDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/ESI/ESIAttributes.cpp.inc"
      >();
}
