//===- SystemCTypesImpl.td - SystemC type implementations --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type implementations for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"

class SystemCTypeDef<string name> : TypeDef<SystemCDialect, name> { }
