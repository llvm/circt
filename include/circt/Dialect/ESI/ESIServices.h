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

class ServiceGeneratorDispatcher {
public:
  using ServiceGeneratorFunc =
      llvm::function_ref<LogicalResult(ServiceImplementReqOp)>;
  using CreateLookupTable =
      llvm::function_ref<DenseMap<Attribute, ServiceGeneratorFunc>(
          MLIRContext *)>;

  ServiceGeneratorDispatcher(CreateLookupTable create, bool failIfNotFound)
      : create(create), failIfNotFound(failIfNotFound) {}
  ServiceGeneratorDispatcher(const ServiceGeneratorDispatcher &that)
      : create(that.create), failIfNotFound(that.failIfNotFound) {}

  static ServiceGeneratorDispatcher defaultDispatcher();

  LogicalResult generate(ServiceImplementReqOp);

private:
  CreateLookupTable create;
  bool failIfNotFound;
  llvm::Optional<DenseMap<Attribute, ServiceGeneratorFunc>> lookupGen;
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createESIConnectServicesPass();

} // namespace esi
} // namespace circt

#endif
