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

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/IndentingOStream.h"
#include "llvm/ADT/MapVector.h"

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

/// Emit an ID in capnp format.
llvm::raw_ostream &emitId(llvm::raw_ostream &os, int64_t id);

/// Every time we implement a breaking change in the schema generation,
/// increment this number. It is a seed for all the schema hashes.
constexpr uint64_t esiCosimSchemaVersion = 1;

namespace detail {
struct CapnpTypeSchemaImpl;
} // namespace detail

// Base type for all Cap'nProto-implementing type emitters.
class ESICapnpType {
public:
  using FieldInfo = hw::StructType::FieldInfo;

  ESICapnpType(mlir::Type);
  bool operator==(const ESICapnpType &) const;

  /// Get the type back.
  mlir::Type getType() const { return type; }

  /// Get the Cap'nProto schema ID for a type.
  uint64_t capnpTypeID() const;

  /// Returns true if the type is currently supported.
  bool isSupported() const;

  /// Get the capnp struct name.
  llvm::StringRef capnpName() const;

  /// Write out the name and ID in capnp schema format.
  void writeMetadata(llvm::raw_ostream &os) const;

  llvm::ArrayRef<FieldInfo> getFields() const { return fieldTypes; }

private:
  /// Capnp requires that everything be contained in a struct. ESI doesn't so
  /// we wrap non-struct types in a capnp struct. During decoder/encoder
  /// construction, it's convenient to use the capnp model so assemble the
  /// virtual list of `Type`s here.
  SmallVector<FieldInfo> fieldTypes;

  mlir::Type type;
  mutable std::string cachedName;
  mutable std::optional<uint64_t> cachedID;
};

/// Generate and reason about a Cap'nProto schema for a particular MLIR type.
class CapnpTypeSchema : public ESICapnpType {
public:
  CapnpTypeSchema(mlir::Type);

  using ESICapnpType::operator==;

  /// Size in bits of the capnp message.
  size_t size() const;

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
  std::shared_ptr<detail::CapnpTypeSchemaImpl> s;

  /// Cache of the decode/encode modules;
  static llvm::SmallDenseMap<Type, hw::HWModuleOp> decImplMods;
  static llvm::SmallDenseMap<Type, hw::HWModuleOp> encImplMods;
};

// Generate and reason about a C++ type for a particular Cap'nProto and MLIR
// type.
class CPPType : public ESICapnpType {
public:
  CPPType(mlir::Type type);

  /// Get the C++ struct name.
  llvm::StringRef cppName() const;

  /// Write out the C++ name of this type.
  void writeCppName(llvm::raw_ostream &os) const;

  /// Write out the type in its entirety.
  mlir::LogicalResult write(support::indenting_ostream &os) const;

  // Emits an RTTR registration for this type. If provided, the `namespace`
  // should indicate the namespace wherein this type was emitted.
  void writeReflection(support::indenting_ostream &os,
                       llvm::ArrayRef<std::string> namespaces) const;
};

struct CPPEndpoint {
  CPPEndpoint(
      esi::ServicePortInfo portInfo,
      const llvm::MapVector<mlir::Type, circt::esi::capnp::CPPType> &types)
      : portInfo(portInfo), types(types) {}
  StringRef getName() const { return portInfo.name.getValue(); }
  std::string getTypeName() const { return "T" + getName().str(); }
  std::string getPointerTypeName() const { return getTypeName() + "Ptr"; }
  LogicalResult writeType(Location loc, support::indenting_ostream &os) const;
  LogicalResult writeDecl(Location loc, support::indenting_ostream &os) const;

  esi::ServicePortInfo portInfo;

  // A mapping of MLIR types to their CPPType counterparts. Ensures consistency
  // between the emitted type signatures and those used in the service endpoint
  // API.
  const llvm::MapVector<mlir::Type, circt::esi::capnp::CPPType> &types;
};

class CPPService {
public:
  CPPService(
      esi::ServiceDeclOpInterface service,
      const llvm::MapVector<mlir::Type, circt::esi::capnp::CPPType> &types);

  // Return the name of this service.
  StringRef name() const {
    return SymbolTable::getSymbolName(service).getValue();
  }

  // Write out the C++ API of this service.
  LogicalResult write(support::indenting_ostream &os);

  esi::ServiceDeclOpInterface getService() { return service; }
  llvm::SmallVector<ServicePortInfo> getPorts();
  CPPEndpoint *getPort(llvm::StringRef portName);

  auto &getEndpoints() { return endpoints; }

private:
  esi::ServiceDeclOpInterface service;

  // Note: cannot use llvm::SmallVector on a forward declared class.
  llvm::SmallVector<std::shared_ptr<CPPEndpoint>> endpoints;
};

class CPPDesignModule {
public:
  CPPDesignModule(hw::HWModuleLike mod,
                  SmallVectorImpl<ServiceHierarchyMetadataOp> &services,
                  llvm::SmallVectorImpl<capnp::CPPService> &cppServices)
      : mod(mod), services(services), cppServices(cppServices) {}

  llvm::StringRef getCPPName() { return mod.getModuleName(); }
  LogicalResult write(support::indenting_ostream &ios);

private:
  hw::HWModuleLike mod;
  SmallVectorImpl<ServiceHierarchyMetadataOp> &services;
  SmallVectorImpl<capnp::CPPService> &cppServices;
};

} // namespace capnp
} // namespace esi
} // namespace circt

#endif // CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H
