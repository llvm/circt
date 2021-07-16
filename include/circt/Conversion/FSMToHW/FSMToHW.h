//===- FSMToHW.h - FSM to HW conversions ------------------*- C++ -*-===//
//
// This file declares passes which will convert the FSM dialect to HW and SV.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FSMTOHW_FSMTOHW_H
#define CIRCT_CONVERSION_FSMTOHW_FSMTOHW_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
std::unique_ptr<mlir::Pass> createConvertFSMToHWPass();
} // namespace circt

#endif // CIRCT_CONVERSION_FSMTOHW_FSMTOHW_H
