//===- spt-opt.cpp - The spt-opt driver -----------------------------------===//
//
// This file implements the 'spt-opt' tool, which is the spt analog of mlir-opt,
// used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"

using namespace llvm;

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "spt modular optimizer driver\n");

  // TODO: Implement stuff! :-)
  return 0;
}
