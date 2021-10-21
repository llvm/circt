#ifndef CIRCT_TOOLS_HLT_STDSIMINTERFACE_H
#define CIRCT_TOOLS_HLT_STDSIMINTERFACE_H

#include <optional>

#include "circt/Tools/hlt/Simulator/SimInterface.h"

namespace circt {
namespace hlt {

template <typename TInput, typename TOutput>
class StdSimInterface : public SimInterface<TInput, TOutput> {
public:
  void step() override {
    if (!inBuffer.has_value())
      return;
    outBuffer = call(inBuffer.value());
    inBuffer.reset();
    ++iterations;
  }

  // The simulator is ready to accept a new input if there isn't an output
  // value in the output buffer that needs to be popped.
  bool inReady() override { return !outBuffer.has_value(); }

  // The simulator output is valid whenever the output buffer has a value.
  bool outValid() override { return outBuffer.has_value(); }
  void pushInput(const TInput &input) override { inBuffer = input; }
  TOutput popOutput() {
    assert(outBuffer.has_value());
    TOutput v = outBuffer.value();
    outBuffer.reset();
    return v;
  }
  void setup() override {}
  void finish() override {}
  uint64_t time() override { return iterations; }
  void dump(std::ostream & /*os*/) const override {}

protected:
  // Function which must be overwritten by the generated simulator; this will
  // map to the 'std' function symbol defined by lowering the std code through
  // LLVMIR.
  virtual TOutput call(const TInput &input) = 0;

private:
  // Nullable in- and output buffers.
  std::optional<TInput> inBuffer;
  std::optional<TOutput> outBuffer;

  // The std simulator defines its timestep as # of times a function has been
  // invoked.
  unsigned iterations = 0;
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_STDSIMINTERFACE_H
