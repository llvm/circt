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

#ifndef CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H
#define CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H

#include "mlir/IR/Types.h"

#include <memory>

namespace circt {
namespace esi {
namespace capnp {

/// Every time we implement a breaking change in the schema generation,
/// increment this number. It is a seed for all the schema hashes.
constexpr uint64_t esiCosimSchemaVersion = 1;

namespace detail {
struct TypeSchemaStorage;
} // namespace detail

class TypeSchema {
public:
  TypeSchema(mlir::Type);

  /// Get the Cap'nProto schema ID for a type.
  uint64_t capnpTypeID() const;

  bool isSupported() const;
  size_t size() const;
  llvm::StringRef name() const;
  mlir::LogicalResult write(llvm::raw_ostream &os);
  mlir::LogicalResult writeMetadata(llvm::raw_ostream &os);

  bool operator==(const TypeSchema &) const;

private:
  std::shared_ptr<detail::TypeSchemaStorage> s;
};

} // namespace capnp
} // namespace esi
} // namespace circt

#endif // CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H
