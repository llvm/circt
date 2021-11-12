//===- StaticLogicDialect.td - StaticLogic dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STATICLOGICDIALECT_TD
#define STATICLOGICDIALECT_TD

include "mlir/IR/OpBase.td"

def StaticLogic_Dialect : Dialect {
  let name = "staticlogic";
  let cppNamespace = "::circt::staticlogic";
  let extraClassDeclaration = [{
    /// Register all StaticLogic attributes.
    void registerAttributes();
  }];
}

#endif // STATICLOGICDIALECT_TD
