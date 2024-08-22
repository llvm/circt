//===- arcilator-jit-env.cpp - Internal arcilator JIT API -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the arcilator JIT runtime environment API facing
// the arcilator.cpp.
//
//===----------------------------------------------------------------------===//

#include "../arcilator-runtime.h"

#define ARCJITENV_EXPORTS
#include "arcilator-jit-env.h"

ARCJITENV_API int arc_jit_runtime_env_init(void) { return 0; }
ARCJITENV_API void arc_jit_runtime_env_deinit(void) { return; }
