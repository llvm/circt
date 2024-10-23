// Auto-generated from gcd.state.json
#include <cstddef>
#include <cstdint>
struct gcd_state {
  struct IO {
    volatile uint8_t _padding_0[25];
    bool i_clk;
    bool i_rst;
    uint64_t i_in_a;
    uint64_t i_in_b;
    bool i_in_valid;
    volatile uint8_t _padding_1[25];
    bool o_in_ready;
    uint64_t o_out;
    uint64_t o_out_valid;
  } io;
  volatile uint8_t _tail[20];
};
static_assert(offsetof(gcd_state, io.i_clk) == 25);
static_assert(offsetof(gcd_state, io.i_rst) == 26);
static_assert(offsetof(gcd_state, io.i_in_a) == 32);
static_assert(offsetof(gcd_state, io.i_in_b) == 40);
static_assert(offsetof(gcd_state, io.i_in_valid) == 48);
static_assert(offsetof(gcd_state, io.o_in_ready) == 74);
static_assert(offsetof(gcd_state, io.o_out) == 80);
static_assert(offsetof(gcd_state, io.o_out_valid) == 88);
