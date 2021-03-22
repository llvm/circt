//===- MSFTAttributescpp - Implement MSFT dialect attributes --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSFT dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace msft;

namespace circt {
namespace msft {
namespace detail {

struct PhysLocationAttrStorage : public mlir::AttributeStorage {
  PhysLocationAttrStorage(DeviceTypeAttr type, uint64_t x, uint64_t y,
                          uint64_t num, StringRef entity)
      : type(type), x(x), y(y), num(num), entity(entity) {}

  using KeyTy =
      std::tuple<DeviceTypeAttr, uint64_t, uint64_t, uint64_t, StringRef>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(type, x, y, num, entity);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key), std::get<3>(key),
                              std::get<4>(key));
  }

  static PhysLocationAttrStorage *
  construct(mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<PhysLocationAttrStorage>())
        PhysLocationAttrStorage(std::get<0>(key), std::get<1>(key),
                                std::get<2>(key), std::get<3>(key),
                                std::get<4>(key));
  }

  DeviceTypeAttr type;
  uint64_t x, y, num;
  StringRef entity;
};

} // namespace detail
} // namespace msft
} // namespace circt

PhysLocationAttr PhysLocationAttr::get(DeviceTypeAttr type, uint64_t x,
                                       uint64_t y, uint64_t num,
                                       StringRef entity) {
  return Base::get(type.getContext(), type, x, y, num, entity);
}

DeviceTypeAttr PhysLocationAttr::Type() const { return getImpl()->type; }
uint64_t PhysLocationAttr::X() const { return getImpl()->x; }
uint64_t PhysLocationAttr::Y() const { return getImpl()->y; }
uint64_t PhysLocationAttr::Num() const { return getImpl()->num; }
StringRef PhysLocationAttr::Entity() const { return getImpl()->entity; }

Attribute PhysLocationAttr::parse(DialectAsmParser &p) {
  llvm::SMLoc loc = p.getCurrentLocation();
  StringRef devTypeStr;
  uint64_t x, y, num;
  StringAttr entity;
  if (p.parseLess() || p.parseKeyword(&devTypeStr) || p.parseComma() ||
      p.parseInteger(x) || p.parseComma() || p.parseInteger(y) ||
      p.parseComma() || p.parseInteger(num) || p.parseComma() ||
      p.parseAttribute(entity) || p.parseGreater())
    return Attribute();

  auto *ctxt = p.getBuilder().getContext();
  Optional<DeviceType> devType = symbolizeDeviceType(devTypeStr);
  if (!devType) {
    p.emitError(loc, "Unknown device type '" + devTypeStr + "'");
    return Attribute();
  }
  DeviceTypeAttr devTypeAttr = DeviceTypeAttr::get(ctxt, *devType);
  auto phy = PhysLocationAttr::get(devTypeAttr, x, y, num, entity.getValue());
  return phy;
}

void PhysLocationAttr::print(DialectAsmPrinter &p) const {
  p << "physloc<" << stringifyDeviceType(Type().getValue()) << ", " << X()
    << ", " << Y() << ", " << Num() << ", \"" << Entity() << "\">";
}

void MSFTDialect::registerAttributes() { addAttributes<PhysLocationAttr>(); }
