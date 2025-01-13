//===- InitAllPasses.h - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all passes to the
// system.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_INITALLPASSES_H_
#define CIRCT_INITALLPASSES_H_

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/FSM/FSMPasses.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/Pipeline/PipelinePasses.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Dialect/SSP/SSPPasses.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Dialect/SystemC/SystemCPasses.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "circt/Transforms/Passes.h"

namespace circt {

inline void registerAllPasses() {
  // Conversion Passes
  registerCIRCTConversionPasses();

  // Transformation passes
  registerCIRCTTransformsPasses();

  // LEC transformation passes
  registerLECTransformsPasses();

  // BMC transformation passes
  registerBMCTransformsPasses();

  // Standard Passes
  aig::registerPasses();
  arc::registerPasses();
  calyx::registerPasses();
  comb::registerPasses();
  dc::registerPasses();
  esi::registerESIPasses();
  firrtl::registerPasses();
  fsm::registerPasses();
  llhd::registerPasses();
  msft::registerPasses();
  om::registerPasses();
  seq::registerPasses();
  sv::registerPasses();
  handshake::registerPasses();
  kanagawa::registerPasses();
  rtg::registerPasses();
  hw::registerPasses();
  pipeline::registerPasses();
  sim::registerPasses();
  ssp::registerPasses();
  systemc::registerPasses();
  verif::registerPasses();
  moore::registerPasses();
}

} // namespace circt

#endif // CIRCT_INITALLPASSES_H_
