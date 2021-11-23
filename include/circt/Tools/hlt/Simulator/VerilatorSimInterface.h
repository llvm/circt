#ifndef CIRCT_TOOLS_HLT_VERILATORSIMINTERFACE_H
#define CIRCT_TOOLS_HLT_VERILATORSIMINTERFACE_H

#include <functional>

#include "circt/Tools/hlt/Simulator/SimDriver.h"

#include "verilated.h"

#if VM_TRACE
#include "verilated_vcd_c.h"
#endif

// Legacy function required only so linking works on Cygwin and MSVC++
double sc_time_stamp() { return 0; }

namespace circt {
namespace hlt {

/// Generic interface to access various parts of the verilated model. This is
/// needed due to verilator models themselves not inheriting from some form of
/// interface.
struct VerilatorGenericInterface {
  CData *clock = nullptr;
  CData *reset = nullptr;
  CData *nReset = nullptr;
};

template <typename TInput, typename TOutput, typename TModel>
class VerilatorSimInterface : public SimInterface<TInput, TOutput> {
public:
  VerilatorSimInterface() : SimInterface<TInput, TOutput>() {
    // Instantiate the verilated model
    ctx = std::make_unique<VerilatedContext>();
    dut = std::make_unique<TModel>(ctx.get());

#if VM_TRACE
    ctx->traceEverOn(true);
    trace = std::make_unique<VerilatedVcdC>();
    // Log 99 levels of hierarchy
    dut->trace(trace.get(), 99);
    // Create logging output directory
    Verilated::mkdir("logs");
    trace->open("logs/vlt_dump.vcd");
#endif
  }

  uint64_t time() override { return ctx->time(); }

  bool outValid() override {
    return std::all_of(this->outPorts.begin(), this->outPorts.end(),
                       [](auto &port) { return port->valid(); });
  }

  void step() override { clock_flip(); }

  void dump(std::ostream &out) const override {
    out << "Port states:\n";
    for (auto &inPort : this->inPorts)
      out << *inPort << "\n";
    for (auto &outPort : this->outPorts)
      out << *outPort << "\n";
    out << "\n";
  }

  void setup() override {
    // Verify generic interface
    assert(interface.clock != nullptr && "Must set pointer to clock signal");
    assert((static_cast<bool>(interface.reset) ^
            static_cast<bool>(interface.nReset)) &&
           "Must set pointer to either reset or nReset");

    // Reset top-level model
    if (interface.reset)
      *interface.reset = !0;
    else
      *interface.nReset = !1;

    // Reset in- and output ports
    for (auto &port : this->inPorts)
      port->reset();
    for (auto &port : this->outPorts)
      port->reset();

    // Run for a few cycles with reset.
    for (int i = 0; i < 2; ++i)
      clock();

    // Disassert reset
    if (interface.reset)
      *interface.reset = !1;
    else
      *interface.nReset = !0;
    clock();
  }

  void finish() override {
    dut->final();

#if VM_TRACE
    // Close trace if opened.
    trace->close();
#endif
  }

protected:
  void advanceTime() {
#if VM_TRACE
    trace->dump(ctx->time());
    // If tracing, flush after each cycle so we can immediately see the output.
    trace->flush();
#endif
    ctx->timeInc(1);
    dut->eval();
  }

  // Clocks the model a half phase (rising or falling edge)
  void clock_half(bool rising) {
    // Ensure combinational logic is settled, if input pins changed.
    advanceTime();
    *interface.clock = rising;
    dut->eval();
    advanceTime();
  }

  void clock_rising() { clock_half(true); }
  void clock_falling() { clock_half(false); }
  void clock_flip() { clock_half(!*interface.clock); }
  void clock() {
    clock_rising();
    clock_falling();
  }

  // Pointer to the verilated model.
  std::unique_ptr<TModel> dut;
  std::unique_ptr<VerilatedContext> ctx;
  VerilatorGenericInterface interface;

#if VM_TRACE
  std::unique_ptr<VerilatedVcdC> trace;
#endif
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_VERILATORSIMINTERFACE_H
