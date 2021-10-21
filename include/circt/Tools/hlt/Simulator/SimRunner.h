#ifndef CIRCT_TOOLS_HLT_SIMRUNNER_H
#define CIRCT_TOOLS_HLT_SIMRUNNER_H

#include <assert.h>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <list>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

#include "circt/Tools/hlt/Simulator/SimInterface.h"

#ifndef HLT_TIMEOUT
// Number of steps without meaningful simulator state changes before exiting
// runner.
#define HLT_TIMEOUT 1000
#endif

namespace circt {
namespace hlt {

template <typename TInput, typename TOutput, typename Sim>
class SimRunner {
  using SimQueuesImpl = SimQueues<TInput, TOutput>;

  struct TimeoutCounter {
    int cntr = 0;
    void reset() { cntr = 0; }
    void inc() { cntr++; }
    bool timedOut() const { return cntr >= HLT_TIMEOUT; }
  };

public:
  SimRunner(SimQueuesImpl &queues) : queues(queues) {
    thread = std::thread(&SimRunner::run, this);
  }

  void wakeup() { notifier.notify_all(); }

  // Runner - simulation executer in separate thread
  void run() {
    sim = std::make_unique<Sim>();
    sim->setup();

    debugOut << "RUNNER: Runner thread started" << std::endl;
    std::mutex lock;
    std::unique_lock<std::mutex> ul(lock);
    while (true) { // todo: fix this
      if (to.timedOut()) {
        raiseTimeoutError();
        break;
      }
      if (preStep()) {
        sim->step();
        to.inc();
        debugOut << "+" << std::endl;
      } else {
        debugOut << "RUNNER: Sleeping..." << std::endl;
        notifier.wait(ul);
        debugOut << "RUNNER: Woke up..." << std::endl;
      }
    }
    sim->finish();
  }

  // Todo(mortbopet): This can/should be optimized to not check atomic queues on
  // each step - way too much locking.
  // Returns true if the model should continue evaluating.
  bool preStep() {
    bool cont = false;
    // Rule 1: If has input transaction and sim is ready to accept input
    if (sim->inReady() && !queues.in.empty()) {
      debugOut << "RUNNER: Pushing input (" << sim->time() << ")" << std::endl;
      sim->pushInput(queues.in.pop());
      to.reset();
      cont |= true;
    }
    // Rule 2: If popping an output from the simulator
    if (sim->outValid()) {
      debugOut << "RUNNER: Popping output" << std::endl;
      queues.out.push(sim->popOutput());
      to.reset();
      cont |= true;
    }

    // Rule 3: If someone is awaiting output then always step
    if (!queues.outReq.empty()) {
      if (!queues.out.empty()) {
        debugOut << "RUNNER: Sending output to waiter" << std::endl;
        queues.outReq.pop().get()->notify_all();
      }
      cont |= true;
    }

    return cont;
  }

  /// Checks the current exception pointer of the runner, and rethrows, if any.
  void checkError() {
    epLock.lock();
    std::exception_ptr epCopy = ep;
    epLock.unlock();
    if (epCopy)
      std::rethrow_exception(epCopy);
  }

private:
  // Awakes anyone currently waiting for an output.
  void awakenAll() {
    queues.outReq.lock.lock();
    for (auto &l : queues.outReq.list)
      l->notify_all();
    queues.outReq.lock.unlock();
  }

  // Sets the exception pointer due to a timeout error.
  void raiseTimeoutError() {
    epLock.lock();
    try {
      std::stringstream ss;
      ss << "Timeout reached! Did " << HLT_TIMEOUT
         << " steps without meaningful simulator state change" << std::endl;
      throw std::runtime_error(ss.str());
    } catch (...) {
      ep = std::current_exception();
    }
    epLock.unlock();
    awakenAll();
  }

  std::thread thread;

  // A condition variable which we use to sleep/awake the runner.
  std::condition_variable notifier;
  std::unique_ptr<Sim> sim;
  SimQueuesImpl &queues;

  std::mutex epLock;
  std::exception_ptr ep;

  // A counter to manage timeout'ing this simulation thread.
  TimeoutCounter to;
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_SIMRUNNER_H
