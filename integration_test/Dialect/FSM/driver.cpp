#include "Vtop.h"
#include "verilated_vcd_c.h"
#include <iostream>

int main(int argc, char **argv) {

  Verilated::commandArgs(argc, argv);
  const std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};
  auto *tb = new Vtop;

  // Setup tracing for ease-of-debugging in case this test eventually fails
  // in CI.
  Verilated::traceEverOn(true);
  VerilatedVcdC *tfp = new VerilatedVcdC;
  tb->trace(tfp, 1);
  tfp->open("sim.vcd");

  // Post-reset start time.
  int t0 = 2;

  for (int i = 0; i < 10; i++) {
    if (i > t0)
      std::cout << "out: " << char('A' + tb->out0) << std::endl;

    // Rising edge
    tb->clk = 1;
    tb->eval();
    tfp->dump(i * 2);

    // Testbench
    tb->rst = i < t0;

    // t0:   Starts in A,
    // t0+1: Default transition to B

    if (i == t0 + 2) {
      // B -> C
      tb->in0 = 1;
      tb->in1 = 1;
    }

    if (i == t0 + 3) {
      // C -> B
      tb->in0 = 0;
      tb->in1 = 0;
    }

    if (i == t0 + 4 || i == t0 + 5) {
      // B -> C, C-> A
      tb->in0 = 1;
      tb->in1 = 1;
    }

    // t0+6: Default transition to B

    // Falling edge
    tb->clk = 0;
    tb->eval();
    tfp->dump(i * 2 + 1);
  }
  tfp->close();

  exit(EXIT_SUCCESS);
}
