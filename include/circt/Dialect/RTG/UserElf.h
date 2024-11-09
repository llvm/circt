//===- UserElf.h - Dialect Interface for User-mode Elf Emission -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectInterface.h"

namespace circt {
namespace rtg {

// Dialects indicate they support user-mode elf emission with this interface.
class UserElfInterface : public mlir::DialectInterface::Base<UserElfInterface> {
public:
  UserElfInterface(Dialect *dialect) : Base(dialect) {}
  virtual void emitElfHeader(llvm::raw_ostream &OS,
                             unsigned numContexts) const = 0;
  virtual void emitElfContextHeader(llvm::raw_ostream &OS,
                                    unsigned ctx) const = 0;
  virtual void emitElfContextFooter(llvm::raw_ostream &OS,
                                    unsigned ctx) const = 0;
  virtual void emitElfFooter(llvm::raw_ostream &OS) const = 0;
};

} // namespace rtg
} // namespace circt
