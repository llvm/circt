//===- arcilator-jit-env.h - Internal arcilator JIT API -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations of the JIT runtime environment API facing the arcilator.cpp.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef _WIN32

#ifdef ARCJITENV_EXPORTS
#define ARCJITENV_API extern "C" __declspec(dllexport)
#else
#define ARCJITENV_API extern "C" __declspec(dllimport)
#endif // ARCJITENV_EXPORTS

#else

#define ARCJITENV_API extern "C" __attribute__((visibility("default")))

#endif // _WIN32

// These don't do anything at the moment. It is still
// required to call them to make sure the library
// is linked and loaded before the JIT engine starts.

ARCJITENV_API int arc_jit_runtime_env_init(void);
ARCJITENV_API void arc_jit_runtime_env_deinit(void);
