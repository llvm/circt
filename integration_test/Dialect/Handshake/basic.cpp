//===- basic.cpp - Verilator software driver ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Vtop.h"

int main(int argc, char **argv) {
  // Construct the simulated module's C++ model.
  auto &dut = *new Vtop();

  // Set input data and control signals low.
  dut.arg0_data = 13;
  dut.arg1_data = 52;
  dut.arg0_valid = 0;
  dut.arg1_valid = 0;
  dut.arg3_ready = 0;
  dut.eval();

  // Confirm valid and ready outputs are not asserted.
  assert(dut.arg3_valid == 0 && "output should not be valid");
  assert(dut.arg0_ready == 0 && "input #0 should not be ready");
  assert(dut.arg1_ready == 0 && "input #1 should not be ready");

  // Set one input to valid.
  dut.arg0_valid = 1;
  dut.eval();

  // Confirm output is still not valid.
  assert(dut.arg3_valid == 0 && "output should not be valid");

  // Set the other input to valid.
  dut.arg1_valid = 1;
  dut.eval();

  // Confirm output is now valid but inputs are still not ready.
  assert(dut.arg3_valid == 1 && "output should be valid");
  assert(dut.arg0_ready == 0 && "input #0 should not be ready");
  assert(dut.arg1_ready == 0 && "input #1 should not be ready");

  // Set the output to ready.
  dut.arg3_ready = 1;
  dut.eval();

  // Confirm the output is still valid and the inputs are ready.
  assert(dut.arg3_valid == 1 && "output should be valid");
  assert(dut.arg0_ready == 1 && "input #0 should not be ready");
  assert(dut.arg1_ready == 1 && "input #1 should not be ready");

  // Confirm the result is correct.
  assert(dut.arg3_data == 65 && "output should be correct");

  // Tell the simulator that we're going to exit. This flushes the output(s) and
  // frees whatever memory may have been allocated.
  dut.final();

  return 0;
}
