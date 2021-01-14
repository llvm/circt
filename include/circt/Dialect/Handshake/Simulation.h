//===- Simulation.h -----------------------------------------------===//
//
// Copyright 2021 The CIRCT Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Functions used to execute a restricted form of the standard dialect, and
// the handshake dialect.

#ifndef CIRCT_DIALECT_HANDSHAKE_SIMULATION_H
#define CIRCT_DIALECT_HANDSHAKE_SIMULATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <string>

bool simulate(llvm::StringRef toplevelFunction,
              llvm::ArrayRef<std::string> inputArgs,
              mlir::OwningModuleRef &module, mlir::MLIRContext &context);

#endif
