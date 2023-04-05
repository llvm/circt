//===- LoopScheduleOps.td - LoopSchedule Op Definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LoopSchedule Ops are defined in tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCIT_LOOP_SCHEDULE_OPS_TD
#define CIRCIT_LOOP_SCHEDULE_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "circt/Dialect/LoopSchedule/LoopSchedule.td"

class LoopScheduleOp<string mnemonic, list<Trait> traits = []> :
  Op<LoopSchedule_Dialect, mnemonic, traits>;

#endif // CIRCIT_LOOP_SCHEDULE_OPS_TD
