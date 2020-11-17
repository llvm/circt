//===- PassDetails.h - FIRRTL pass class details ----------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_FIRRTL_PASSDETAILS_H
#define DIALECT_FIRRTL_PASSDETAILS_H

#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/RTL/Ops.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {

#define GEN_PASS_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLPasses.h.inc"

} // namespace firrtl
} // namespace circt

#endif // DIALECT_FIRRTL_PASSDETAILS_H
