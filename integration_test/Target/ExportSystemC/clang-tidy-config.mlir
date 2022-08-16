// REQUIRES: clang-tidy
// RUN: clang-tidy --dump-config %s | FileCheck %s

// CHECK: Checks: 'clang-diagnostic-*,clang-analyzer-*,misc-*,-misc-unused-parameters,-misc-non-private-member-variables-in-classes,readability-identifier-naming'
