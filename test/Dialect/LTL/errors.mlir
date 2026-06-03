// RUN: circt-opt %s --split-input-file --verify-diagnostics

%s = unrealized_conversion_cast to !ltl.sequence

// expected-error @+1 {{attribute 'delay' failed to satisfy constraint}}
ltl.delay %s, -1 : !ltl.sequence

// -----

%s = unrealized_conversion_cast to !ltl.sequence

// expected-error @+1 {{attribute 'length' failed to satisfy constraint}}
ltl.delay %s, 1, -1 : !ltl.sequence

// -----

%s = unrealized_conversion_cast to !ltl.sequence
%clk = hw.constant true

// expected-error @+1 {{attribute 'delay' failed to satisfy constraint}}
ltl.clocked_delay %s, posedge %clk, -1 : !ltl.sequence

// -----

%s = unrealized_conversion_cast to !ltl.sequence
%clk = hw.constant true

// expected-error @+1 {{attribute 'length' failed to satisfy constraint}}
ltl.clocked_delay %s, posedge %clk, 1, -1 : !ltl.sequence
