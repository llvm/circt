//===- JITBind.h - ArcRuntime JIT symbol binding helper -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the helper struct used to bind the IR interface to the MLIR
// Execution Engine without relying on the linker.
//
// This file is specific to the runtime implementation statically linked into
// the arcilator tool and not part of the runtime's API.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_JITBIND_H
#define CIRCT_DIALECT_ARC_RUNTIME_JITBIND_H

namespace circt {
namespace arc {
namespace runtime {

struct APICallbacks {
  uint8_t *(*fnAllocInstance)(const ArcRuntimeModelInfo *model,
                              const char *args);
  void (*fnDeleteInstance)(uint8_t *simState);
  void (*fnOnEval)(uint8_t *simState);
  void (*fnFormat)(const FmtDescriptor *fmt, ...);

  static constexpr char symNameAllocInstance[] = "arcRuntimeIR_allocInstance";
  static constexpr char symNameDeleteInstance[] = "arcRuntimeIR_deleteInstance";
  static constexpr char symNameOnEval[] = "arcRuntimeIR_onEval";
  static constexpr char symNameFormat[] = "arcRuntimeIR_format";
};

#ifdef ARC_RUNTIME_JITBIND_FNDECL
const APICallbacks &getArcRuntimeAPICallbacks();
#endif // ARC_RUNTIME_JITBIND_FNDECL

} // namespace runtime
} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_RUNTIME_JITBIND_H
