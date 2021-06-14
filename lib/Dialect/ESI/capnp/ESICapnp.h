//===- ESICapnp.h - ESI Cap'nProto library utilies --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI utility code which requires libcapnp and libcapnpc.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H
#define CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H

#include "circt/Dialect/HW/HWOps.h"

#include <memory>

namespace mlir {
class Type;
struct LogicalResult;
class Value;
class OpBuilder;
} // namespace mlir
namespace llvm {
class raw_ostream;
class StringRef;
} // namespace llvm

namespace circt {
namespace esi {
namespace capnp {

/// Every time we implement a breaking change in the schema generation,
/// increment this number. It is a seed for all the schema hashes.
constexpr uint64_t esiCosimSchemaVersion = 1;

namespace detail {
struct TypeSchemaImpl;
} // namespace detail

/// Generate and reason about a Cap'nProto schema for a particular MLIR type.
class TypeSchema {
public:
  TypeSchema(mlir::Type);
  bool operator==(const TypeSchema &) const;

  /// Get the type back.
  mlir::Type getType() const;

  /// Get the Cap'nProto schema ID for a type.
  uint64_t capnpTypeID() const;

  /// Returns true if the type is currently supported.
  bool isSupported() const;

  /// Size in bits of the capnp message.
  size_t size() const;

  /// Get the capnp struct name.
  llvm::StringRef name() const;

  /// Write out the name and ID in capnp schema format.
  void writeMetadata(llvm::raw_ostream &os) const;

  /// Write out the schema in its entirety.
  mlir::LogicalResult write(llvm::raw_ostream &os) const;

  /// Build an HW/SV dialect capnp encoder for this type.
  mlir::Value buildEncoder(mlir::OpBuilder &, mlir::Value clk,
                           mlir::Value valid, mlir::Value rawData) const;
  /// Build an HW/SV dialect capnp decoder for this type.
  mlir::Value buildDecoder(mlir::OpBuilder &, mlir::Value clk,
                           mlir::Value valid, mlir::Value capnpData) const;

private:
  /// The implementation of this. Separate to hide the details and avoid having
  /// to include the capnp headers in this header.
  std::shared_ptr<detail::TypeSchemaImpl> s;

  /// Cache of the decode/encode modules;
  static llvm::SmallDenseMap<Type, hw::HWModuleOp> decImplMods;
  static llvm::SmallDenseMap<Type, hw::HWModuleOp> encImplMods;
};

} // namespace capnp
} // namespace esi
} // namespace circt

#endif // CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H
