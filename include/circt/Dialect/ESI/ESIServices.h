//===- ESIServices.h - Code related to ESI services -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESISERVICES_H
#define CIRCT_DIALECT_ESI_ESISERVICES_H

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt {
namespace esi {

/// Class which "dispatches" a service implementation request to its specified
/// generator. Also serves as a registery for generators.
class ServiceGeneratorDispatcher {
public:
  // All generators must support this function pointer signature.
  using ServiceGeneratorFunc =
      llvm::function_ref<LogicalResult(ServiceImplementReqOp)>;

  // Since passes don't have access to a context at creation time (and
  // Attributes are tied to the context), we need to delay lookup table creation
  // until running the dispatch the first time. This is the function pointer
  // signature to create and return that lookup table.
  using CreateLookupTable =
      llvm::function_ref<DenseMap<Attribute, ServiceGeneratorFunc>(
          MLIRContext *)>;

  ServiceGeneratorDispatcher(CreateLookupTable create, bool failIfNotFound)
      : create(create), failIfNotFound(failIfNotFound) {}
  ServiceGeneratorDispatcher(const ServiceGeneratorDispatcher &that)
      : create(that.create), failIfNotFound(that.failIfNotFound) {}

  /// Assemble the a default dispatcher.
  static ServiceGeneratorDispatcher defaultDispatcher();

  /// Generate a service implementation if a generator exists in this registry.
  /// If one is not found, return failure if the `failIfNotFound` flag is set.
  LogicalResult generate(ServiceImplementReqOp);

private:
  CreateLookupTable create;
  bool failIfNotFound;
  llvm::Optional<DenseMap<Attribute, ServiceGeneratorFunc>> lookupTable;
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createESIConnectServicesPass();

} // namespace esi
} // namespace circt

#endif
