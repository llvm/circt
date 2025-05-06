//===- driver.cpp - ESI Verilator software driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A fairly standard, boilerplate Verilator C++ simulation driver. Assumes the
// top level exposes just two signals: 'clk' and 'rst'.
//
//===----------------------------------------------------------------------===//

#ifndef TOP_MODULE
#define TOP_MODULE ESI_Cosim_Top
#endif // TOP_MODULE

// Macro black magic to get the header file name and class name from the
// TOP_MODULE macro. Need to disable formatting for this section, as
// clang-format messes it up by inserting spaces.

// clang-format off
#define STRINGIFY_MACRO(x) STR(x)
#define STR(x) #x
#define EXPAND(x)x
#define CONCAT3(n1, n2, n3) STRINGIFY_MACRO(EXPAND(n1)EXPAND(n2)EXPAND(n3))
#define TOKENPASTE(x, y) x ## y
#define CLASSNAME(x, y) TOKENPASTE(x, y)

#include CONCAT3(V,TOP_MODULE,.h)
// clang-format on

#include "verilated_vcd_c.h"

#include "signal.h"
#include <iostream>
#include <thread>

vluint64_t timeStamp;

// Stop the simulation gracefully on ctrl-c.
volatile bool stopSimulation = false;
void handle_sigint(int) { stopSimulation = true; }

// Called by $time in Verilog.
double sc_time_stamp() { return timeStamp; }

int main(int argc, char **argv) {
  // Register graceful exit handler.
  signal(SIGINT, handle_sigint);

  Verilated::commandArgs(argc, argv);

  // Construct the simulated module's C++ model.
  auto &dut = *new CLASSNAME(V, TOP_MODULE)();
  char *waveformFile = getenv("SAVE_WAVE");

  char *periodStr = getenv("DEBUG_PERIOD");
  unsigned debugPeriod = 0;
  if (periodStr) {
    debugPeriod = std::stoi(periodStr);
    std::cout << "[driver] Setting debug period to " << debugPeriod
              << std::endl;
  }

#ifdef TRACE
  VerilatedVcdC *tfp = nullptr;
#endif

  if (waveformFile) {
#ifdef TRACE
    tfp = new VerilatedVcdC();
    Verilated::traceEverOn(true);
    dut.trace(tfp, 99); // Trace 99 levels of hierarchy
    tfp->open(waveformFile);
    std::cout << "[driver] Writing trace to " << waveformFile << std::endl;
#else
    std::cout
        << "[driver] Warning: waveform file specified, but not a debug build"
        << std::endl;
#endif
  }

  std::cout << "[driver] Starting simulation" << std::endl;

  // TODO: Add max speed (cycles per second) option for small, interactive
  // simulations to reduce waveform for debugging. Should this be a command line
  // option or configurable over the cosim interface?

  // Reset.
  dut.rst = 1;
  dut.clk = 0;

  // TODO: Support ESI reset handshake in the future.
  // Run for a few cycles with reset held.
  for (timeStamp = 0; timeStamp < 8 && !Verilated::gotFinish(); timeStamp++) {
    dut.eval();
    dut.clk = !dut.clk;
#ifdef TRACE
    if (tfp)
      tfp->dump(timeStamp);
#endif
  }

  // Take simulation out of reset.
  dut.rst = 0;

  // Run for the specified number of cycles out of reset.
  for (; !Verilated::gotFinish() && !stopSimulation; timeStamp++) {
    dut.eval();
    dut.clk = !dut.clk;

#ifdef TRACE
    if (tfp)
      tfp->dump(timeStamp);
#endif
    if (debugPeriod)
      std::this_thread::sleep_for(std::chrono::milliseconds(debugPeriod));
  }

  // Tell the simulator that we're going to exit. This flushes the output(s) and
  // frees whatever memory may have been allocated.
  dut.final();
#ifdef TRACE
  if (tfp)
    tfp->close();
#endif

  std::cout << "[driver] Ending simulation at tick #" << timeStamp << std::endl;
  return 0;
}
