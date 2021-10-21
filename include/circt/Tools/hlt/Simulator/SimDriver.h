#ifndef CIRCT_TOOLS_HLT_SIMDRIVER_H
#define CIRCT_TOOLS_HLT_SIMDRIVER_H

#include <assert.h>
#include <chrono>
#include <condition_variable>
#include <list>
#include <memory>
#include <thread>
#include <vector>

#include "circt/Tools/hlt/Simulator/SimInterface.h"
#include "circt/Tools/hlt/Simulator/SimRunner.h"

//===----------------------------------------------------------------------===//
// Sim driver
//===----------------------------------------------------------------------===//

namespace circt {
namespace hlt {

template <typename TInput, typename TOutput, typename Sim>
class SimDriver {
  using SimQueuesImpl = SimQueues<TInput, TOutput>;
  using SimRunnerImpl = SimRunner<TInput, TOutput, Sim>;

  static_assert(std::is_base_of<SimInterface<TInput, TOutput>, Sim>::value,
                "Invalid simulator for this sim driver");

  static_assert(is_instance_of_template<TInput, std::tuple>::value,
                "TInput must be a tuple");
  static_assert(is_instance_of_template<TOutput, std::tuple>::value,
                "TOutput must be a tuple");

public:
  SimDriver() { runner = std::make_unique<SimRunnerImpl>(queues); }

  /// Non-blocking
  void push(const TInput &in) {
    runner->checkError();
    debugOut << "DRIVER: Pushing input..." << std::endl;
    queues.in.push(in);
    runner->wakeup();
  }

  /// Blocking
  TOutput pop() {
    runner->checkError();

    debugOut << "DRIVER: Awaiting output..." << std::endl;
    auto cv = std::make_shared<std::condition_variable>();
    queues.outReq.push(cv);
    runner->wakeup();

    // Todo: there's a race condition between notifying the runner and this
    // thread spinning on uslock.

    // Spin
    std::mutex lock;
    std::unique_lock<std::mutex> ulock(lock);
    cv->wait(ulock);

    // Must check error again in case this thread was awaken due to a simulation
    // error.
    runner->checkError();
    debugOut << "DRIVER: Wakeup, popping output..." << std::endl;

    // Get output
    return queues.out.pop();
  }

private:
  SimQueuesImpl queues;
  std::unique_ptr<SimRunnerImpl> runner;
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_SIMDRIVER_H
