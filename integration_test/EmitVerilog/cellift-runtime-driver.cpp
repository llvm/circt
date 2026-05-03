#include "Vtop.h"
#include "verilated.h"

#include <cstdlib>
#include <iostream>

static bool checkCase(Vtop *dut, const char *name, unsigned a, unsigned aT,
                      unsigned b, unsigned bT, unsigned expectedSum,
                      unsigned expectedTaint) {
  dut->a = a;
  dut->a_t = aT;
  dut->b = b;
  dut->b_t = bT;
  dut->eval();

  auto sum = static_cast<unsigned>(dut->sum & 0xfu);
  auto sumTaint = static_cast<unsigned>(dut->sum_t & 0xfu);
  std::cout << name << " sum=" << sum << " taint=" << sumTaint << std::endl;

  if (sum != expectedSum || sumTaint != expectedTaint) {
    std::cerr << "mismatch for " << name << ": expected sum=" << expectedSum
              << " taint=" << expectedTaint << ", got sum=" << sum
              << " taint=" << sumTaint << std::endl;
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  auto *dut = new Vtop;

  bool ok = true;
  ok &= checkCase(dut, "case0", 1, 0, 2, 0, 3, 0);
  ok &= checkCase(dut, "case1", 3, 1, 1, 0, 4, 7);
  ok &= checkCase(dut, "case2", 0, 0, 0, 15, 0, 15);

  delete dut;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
