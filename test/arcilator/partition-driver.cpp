#include <atomic>
#include <iostream>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include "partition.state.h"

extern "C" void gcd_0_eval(gcd_state *);
extern "C" void gcd_1_eval(gcd_state *);
extern "C" void gcd_0_sync_eval(gcd_state *);
extern "C" void gcd_1_sync_eval(gcd_state *);
extern "C" void gcd_output_eval(gcd_state *);

void dump_state(gcd_state &state) {}

uint16_t gcd_std(uint16_t a, uint16_t b) {
  while (true) {
    uint16_t tmp = b;
    b = a % b;
    a = tmp;

    if (b == 0)
      return tmp;
  }
}

const size_t TICK_UNTIL = 10000000;
std::mt19937 gen(0x19260817);
std::uniform_int_distribution<uint16_t> dist(1, 0xFFFF);

using case_t = std::tuple<uint16_t, uint16_t, uint16_t>;
case_t generate_case() {
  uint16_t a = dist(gen);
  uint16_t b = dist(gen);

  return std::make_tuple(a, b, gcd_std(a, b));
}

int main() {
  gcd_state state;

  std::optional<case_t> last;
  case_t testcase = generate_case();

  std::atomic<size_t> ticket = 0;
  std::vector<std::thread> threads;
  for (int i = 0; i < 2; ++i) {
    threads.emplace_back([&, i]() {
      size_t tick = 0;
      bool leader = i == 0;
      while (tick < TICK_UNTIL) {
        if (leader) {
          state.io.i_clk = tick % 2 == 0;
          state.io.i_rst = tick < 10;

          auto &[a, b, r] = testcase;
          state.io.i_in_a = a;
          state.io.i_in_b = b;
          state.io.i_in_valid = tick > 20;

          // o_in_ready doesn't depend on input, so we can skip a eval here
          // fire and is posedge
          if (tick > 20 && state.io.o_in_ready && tick % 2 == 0) {
            std::cout << "Accept " << a << ", " << b << std::endl;
            if (last) {
              std::cerr << "Last gcd not completed" << std::endl;
              std::exit(1);
            }
            last = testcase;
            testcase = generate_case();
          }
        }

        if (i == 0) {
          size_t cur_ticket = ticket.fetch_add(1, std::memory_order_acq_rel);
          while (cur_ticket < tick * 6 + 2)
            cur_ticket = ticket.load(std::memory_order_acquire);

          gcd_0_eval(&state);
          cur_ticket = ticket.fetch_add(1, std::memory_order_acq_rel);
          while (cur_ticket < tick * 6 + 4)
            cur_ticket = ticket.load(std::memory_order_acquire);

          gcd_0_sync_eval(&state);
          cur_ticket = ticket.fetch_add(1, std::memory_order_acq_rel);
          while (cur_ticket < tick * 6 + 6)
            cur_ticket = ticket.load(std::memory_order_acquire);
        } else {
          size_t cur_ticket = ticket.fetch_add(1, std::memory_order_acq_rel);
          while (cur_ticket < tick * 6 + 2)
            cur_ticket = ticket.load(std::memory_order_acquire);

          gcd_1_eval(&state);
          cur_ticket = ticket.fetch_add(1, std::memory_order_acq_rel);
          while (cur_ticket < tick * 6 + 4)
            cur_ticket = ticket.load(std::memory_order_acquire);

          gcd_1_sync_eval(&state);
          cur_ticket = ticket.fetch_add(1, std::memory_order_acq_rel);
          while (cur_ticket < tick * 6 + 6)
            cur_ticket = ticket.load(std::memory_order_acquire);
        }

        if (leader)
          gcd_output_eval(&state);

        if (leader && tick >= 15 && tick % 2 == 0 && state.io.o_out_valid) {
          // New output generated
          if (!last) {
            std::cerr << "Unexpected valid output" << std::endl;
            std::exit(1);
          }

          auto &[la, lb, lr] = *last;
          std::cout << la << ", " << lb << " -> " << state.io.o_out << " # "
                    << lr << std::endl;

          if (lr != state.io.o_out) {
            std::cerr << "Mismatched output" << std::endl;
            std::exit(1);
          }

          last = {};
        }
        ++tick;
      }
    });
  }

  for (auto &t : threads)
    t.join();
}
