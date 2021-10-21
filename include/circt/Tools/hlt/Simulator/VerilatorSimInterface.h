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

struct VerilatorPort : SimBase {
  virtual void reset() = 0;
  virtual ~VerilatorPort() = default;
};

struct VerilatorInPort : public VerilatorPort {
  virtual bool ready() = 0;
};

struct VerilatorOutPort : public VerilatorPort {
  virtual bool valid() = 0;
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

  bool inReady() override {
    return std::all_of(inPorts.begin(), inPorts.end(),
                       [](auto &port) { return port->ready(); });
  }

  bool outValid() override {
    return std::all_of(outPorts.begin(), outPorts.end(),
                       [](auto &port) { return port->valid(); });
  }

  void step() override { clock(); }

  void dump(std::ostream &out) const override {
    out << "Port states:\n";
    for (auto &inPort : inPorts)
      out << *inPort << "\n";
    for (auto &outPort : outPorts)
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
    for (auto &port : inPorts)
      port->reset();
    for (auto &port : outPorts)
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

  template <typename T, typename... Args>
  T *addInputPort(Args... args) {
    static_assert(std::is_base_of<VerilatorInPort, T>::value,
                  "Port must inherit from VerilatorInPort");
    auto ptr = new T(args...);
    inPorts.push_back(std::move(std::unique_ptr<VerilatorInPort>(ptr)));
    return ptr;
  }

  template <typename T, typename... Args>
  T *addOutputPort(Args... args) {
    static_assert(std::is_base_of<VerilatorOutPort, T>::value,
                  "Port must inherit from VerilatorOutPort");
    auto ptr = new T(args...);
    outPorts.push_back(std::move(std::unique_ptr<VerilatorOutPort>(ptr)));
    return ptr;
  }

protected:
  // Clocks the model.
  void clock() {
    // Ensure combinational logic is settled, if input pins changed.
    *interface.clock = 0;
    dut->eval();

    // Rising edge
    *interface.clock = 1;
    dut->eval();
    ctx->timeInc(1);
#if VM_TRACE
    trace->dump(ctx->time());
#endif
    // Falling edge
    *interface.clock = 0;
    dut->eval();
    ctx->timeInc(1);
#if VM_TRACE
    trace->dump(ctx->time());
    // If tracing, flush after each cycle so we can immediately see the output.
    trace->flush();
#endif
  }

  // Pointer to the verilated model.
  std::unique_ptr<TModel> dut;
  std::unique_ptr<VerilatedContext> ctx;
  std::vector<std::unique_ptr<VerilatorInPort>> inPorts;
  std::vector<std::unique_ptr<VerilatorOutPort>> outPorts;
  VerilatorGenericInterface interface;

#if VM_TRACE
  std::unique_ptr<VerilatedVcdC> trace;
#endif
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_VERILATORSIMINTERFACE_H
