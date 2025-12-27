// RUN: circt-opt --sim-proceduralize --canonicalize %s | FileCheck %s

// CHECK-LABEL: @basic_print1
// CHECK-NEXT:  %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT:  hw.triggered posedge %[[TRG]] {
// CHECK-NEXT:     %[[LIT:.*]] = sim.fmt.literal "Test"
// CHECK-NEXT:     sim.proc.print %[[LIT]]
// CHECK-NEXT:   }

hw.module @basic_print1(in %clk : !seq.clock) {
  %true = hw.constant true
  %test = sim.fmt.literal "Test"
  sim.print %test on %clk if %true
}

// CHECK-LABEL: @basic_print2
// CHECK-NEXT:  %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT:  hw.triggered posedge %[[TRG]](%cond) : i1 {
// CHECK-NEXT:  ^bb0(%[[ARG:.*]]: i1):
// CHECK-DAG:      %[[LIT1:.*]] = sim.fmt.literal "Not with a bang but a \00"
// CHECK-DAG:      %[[LIT0:.*]] = sim.fmt.literal "This is the way the world ends\0A"
// CHECK:          scf.if %[[ARG]] {
// CHECK-NEXT:       sim.proc.print %[[LIT0]]
// CHECK-NEXT:       sim.proc.print %[[LIT0]]
// CHECK-NEXT:       sim.proc.print %[[LIT0]]
// CHECK-NEXT:       sim.proc.print %[[LIT1]]
// CHECK-NEXT:     }
// CHECK-NEXT:   }

hw.module @basic_print2(in %clk : !seq.clock, in %cond : i1) {
  %0 = sim.fmt.literal "Not with a bang but a \00"
  %1 = sim.fmt.literal "This is the way the world ends\0A"

  sim.print %1 on %clk if %cond
  sim.print %1 on %clk if %cond
  sim.print %1 on %clk if %cond
  sim.print %0 on %clk if %cond
}

// CHECK-LABEL: @basic_print3
// CHECK-NEXT:  %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT:  hw.triggered posedge %[[TRG]](%val) : i32 {
// CHECK-NEXT:  ^bb0(%[[ARG:.*]]: i32):
// CHECK-DAG:      %[[LB:.*]] = sim.fmt.literal "Bin: "
// CHECK-DAG:      %[[LD:.*]] = sim.fmt.literal ", Dec: "
// CHECK-DAG:      %[[LH:.*]] = sim.fmt.literal ", Hex: "
// CHECK-DAG:      %[[LO:.*]] = sim.fmt.literal ", Oct: "
// CHECK-DAG:      %[[FB:.*]] = sim.fmt.bin %[[ARG]] {specifierWidth = 32 : i32} : i32
// CHECK-DAG:      %[[FD:.*]] = sim.fmt.dec %[[ARG]] {specifierWidth = 10 : i32} : i32
// CHECK-DAG:      %[[FH:.*]] = sim.fmt.hex %[[ARG]], isUpper false {specifierWidth = 8 : i32} : i32
// CHECK-DAG:      %[[FO:.*]] = sim.fmt.oct %[[ARG]] {specifierWidth = 11 : i32} : i32
// CHECK-DAG:      %[[CAT:.*]] = sim.fmt.concat (%[[LB]], %[[FB]], %[[LD]], %[[FD]], %[[LH]], %[[FH]], %[[LO]], %[[FO]])
// CHECK:          sim.proc.print %[[CAT]]
// CHECK-NEXT:   }

hw.module @basic_print3(in %clk : !seq.clock, in %val: i32) {
  %true = hw.constant true
  %comma = sim.fmt.literal ", "

  %bin_lit = sim.fmt.literal "Bin: "
  %bin_val = sim.fmt.bin %val {specifierWidth = 32 : i32} : i32
  %bin_cat = sim.fmt.concat (%bin_lit, %bin_val)

  %dec_lit = sim.fmt.literal "Dec: "
  %dec_val = sim.fmt.dec %val {specifierWidth = 10 : i32} : i32
  %dec_cat = sim.fmt.concat (%dec_lit, %dec_val)

  %hex_lit = sim.fmt.literal "Hex: "
  %hex_val = sim.fmt.hex %val, isUpper false {specifierWidth = 8 : i32} : i32
  %hex_cat = sim.fmt.concat (%hex_lit, %hex_val)

  %oct_lit = sim.fmt.literal "Oct: "
  %oct_val = sim.fmt.oct %val {specifierWidth = 11 : i32} : i32
  %oct_cat = sim.fmt.concat (%oct_lit, %oct_val)

  %str = sim.fmt.concat (%bin_cat, %comma, %dec_cat, %comma, %hex_cat, %comma, %oct_cat)

  sim.print %str on %clk if %true
}

// CHECK-LABEL: @multi_args
// CHECK-NEXT:  %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT:  hw.triggered posedge %0(%a, %b, %c) : i8, i8, i8 {
// CHECK-NEXT:  ^bb0(%[[ARG0:.*]]: i8, %[[ARG1:.*]]: i8, %[[ARG2:.*]]: i8):
// CHECK-DAG:      %[[COM:.*]] = sim.fmt.literal ", "
// CHECK-DAG:      %[[B0:.*]] = sim.fmt.bin %[[ARG0]] {specifierWidth = 8 : i32} : i8
// CHECK-DAG:      %[[H0:.*]] = sim.fmt.hex %[[ARG0]], isUpper false {specifierWidth = 2 : i32} : i8
// CHECK-DAG:      %[[O0:.*]] = sim.fmt.oct %[[ARG0]] {specifierWidth = 3 : i32} : i8
// CHECK-DAG:      %[[B1:.*]] = sim.fmt.bin %[[ARG1]] {specifierWidth = 8 : i32} : i8
// CHECK-DAG:      %[[H1:.*]] = sim.fmt.hex %[[ARG1]], isUpper false {specifierWidth = 2 : i32} : i8
// CHECK-DAG:      %[[O1:.*]] = sim.fmt.oct %[[ARG1]] {specifierWidth = 3 : i32} : i8
// CHECK-DAG:      %[[B2:.*]] = sim.fmt.bin %[[ARG2]] {specifierWidth = 8 : i32} : i8
// CHECK-DAG:      %[[H2:.*]] = sim.fmt.hex %[[ARG2]], isUpper false {specifierWidth = 2 : i32} : i8
// CHECK-DAG:      %[[O2:.*]] = sim.fmt.oct %[[ARG2]] {specifierWidth = 3 : i32} : i8
// CHECK-DAG:      %[[CAT:.*]] = sim.fmt.concat (%[[B0]], %[[B1]], %[[B2]], %[[COM]], %[[H0]], %[[H1]], %[[H2]], %[[COM]], %[[O0]], %[[O1]], %[[O2]])
// CHECK:          sim.proc.print %[[CAT]]
// CHECK-NEXT:   }

hw.module @multi_args(in %clk : !seq.clock, in %a: i8, in %b: i8, in %c: i8) {
  %true = hw.constant true
  %comma = sim.fmt.literal ", "

  %bina = sim.fmt.bin %a {specifierWidth = 8 : i32} : i8
  %binb = sim.fmt.bin %b {specifierWidth = 8 : i32} : i8
  %binc = sim.fmt.bin %c {specifierWidth = 8 : i32} : i8

  %hexa = sim.fmt.hex %a, isUpper false {specifierWidth = 2 : i32} : i8
  %hexb = sim.fmt.hex %b, isUpper false {specifierWidth = 2 : i32} : i8
  %hexc = sim.fmt.hex %c, isUpper false {specifierWidth = 2 : i32} : i8

  %octa = sim.fmt.oct %a {specifierWidth = 3 : i32} : i8
  %octb = sim.fmt.oct %b {specifierWidth = 3 : i32} : i8
  %octc = sim.fmt.oct %c {specifierWidth = 3 : i32} : i8

  %cat = sim.fmt.concat (%bina, %binb, %binc, %comma, %hexa, %hexb, %hexc, %comma, %octa, %octb, %octc)

  sim.print %cat on %clk if %true
}

// CHECK-LABEL: @multi_clock
// CHECK-NEXT:  %[[TRGA:.*]] = seq.from_clock %clka
// CHECK-NEXT:  hw.triggered posedge %[[TRGA]](%val) : i32 {
// CHECK-NEXT:  ^bb0(%[[ARGA:.*]]: i32):
// CHECK-DAG:      %[[LA0:.*]] = sim.fmt.literal "Val is 0x"
// CHECK-DAG:      %[[LA1:.*]] = sim.fmt.literal " on A."
// CHECK-DAG:      %[[FA:.*]] = sim.fmt.hex %[[ARGA]], isUpper false {specifierWidth = 8 : i32} : i32
// CHECK-DAG:      %[[CATA:.*]] = sim.fmt.concat (%[[LA0]], %[[FA]], %[[LA1]])
// CHECK:          sim.proc.print %[[CATA]]
// CHECK-NEXT:   }
// CHECK-NEXT:  %[[TRGB:.*]] = seq.from_clock %clkb
// CHECK-NEXT:  hw.triggered posedge %[[TRGB]](%val) : i32 {
// CHECK-NEXT:  ^bb0(%[[ARGB:.*]]: i32):
// CHECK-DAG:      %[[LB0:.*]] = sim.fmt.literal "Val is 0x"
// CHECK-DAG:      %[[LB1:.*]] = sim.fmt.literal " on B."
// CHECK-DAG:      %[[FB:.*]] = sim.fmt.hex %[[ARGB]], isUpper false {specifierWidth = 8 : i32} : i32
// CHECK-DAG:      %[[CATB:.*]] = sim.fmt.concat (%[[LB0]], %[[FB]], %[[LB1]])
// CHECK:          sim.proc.print %[[CATB]]
// CHECK-NEXT:   }
// CHECK-NEXT:  %[[TRGC:.*]] = seq.from_clock %clkc
// CHECK-NEXT:  hw.triggered posedge %[[TRGC]](%val) : i32 {
// CHECK-NEXT:  ^bb0(%[[ARGC:.*]]: i32):
// CHECK-DAG:      %[[LC0:.*]] = sim.fmt.literal "Val is 0x"
// CHECK-DAG:      %[[LC1:.*]] = sim.fmt.literal " on C."
// CHECK-DAG:      %[[FC:.*]] = sim.fmt.hex %[[ARGC]], isUpper false {specifierWidth = 8 : i32} : i32
// CHECK-DAG:      %[[CATC:.*]] = sim.fmt.concat (%[[LC0]], %[[FC]], %[[LC1]])
// CHECK:          sim.proc.print %[[CATC]]
// CHECK-NEXT:   }

hw.module @multi_clock(in %clka : !seq.clock, in %clkb : !seq.clock, in %clkc : !seq.clock, in %val: i32) {
  %true = hw.constant true
  %pre = sim.fmt.literal "Val is 0x"
  %hex_val = sim.fmt.hex %val, isUpper false {specifierWidth = 8 : i32} : i32

  %onA = sim.fmt.literal " on A."
  %onB = sim.fmt.literal " on B."
  %onC = sim.fmt.literal " on C."

  %catA = sim.fmt.concat (%pre, %hex_val, %onA)
  sim.print %catA on %clka if %true

  %catB = sim.fmt.concat (%pre, %hex_val, %onB)
  sim.print %catB on %clkb if %true

  %catC = sim.fmt.concat (%pre, %hex_val, %onC)
  sim.print %catC on %clkc if %true
}

// CHECK-LABEL: @sequence
// CHECK-NEXT:  %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT:  hw.triggered posedge %[[TRG]](%conda, %condb, %val) : i1, i1, i32 {
// CHECK-NEXT:  ^bb0(%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: i32):
// CHECK-DAG:      %[[L1:.*]] = sim.fmt.literal "#1"
// CHECK-DAG:      %[[L2:.*]] = sim.fmt.literal "#2"
// CHECK-DAG:      %[[L3:.*]] = sim.fmt.literal "#3"
// CHECK-DAG:      %[[L4:.*]] = sim.fmt.literal "#4"
// CHECK-DAG:      %[[L5:.*]] = sim.fmt.literal "#5"
// CHECK-DAG:      %[[L6:.*]] = sim.fmt.literal "#6"
// CHECK-DAG:      %[[BIN:.*]] = sim.fmt.bin %[[ARG2]] {specifierWidth = 32 : i32} : i32
// CHECK:          scf.if %[[ARG0]] {
// CHECK-NEXT:       sim.proc.print %[[L1]]
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.if %[[ARG1]] {
// CHECK-NEXT:       sim.proc.print %[[L2]]
// CHECK-NEXT:       sim.proc.print %[[L3]]
// CHECK-NEXT:       sim.proc.print %[[L4]]
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.if %[[ARG0]] {
// CHECK-NEXT:       sim.proc.print %[[L5]]
// CHECK-NEXT:     }
// CHECK-NEXT:     sim.proc.print %[[BIN]]
// CHECK-NEXT:     scf.if %[[ARG0]] {
// CHECK-NEXT:       sim.proc.print %[[L6]]
// CHECK-NEXT:     }
// CHECK-NEXT:   }

hw.module @sequence(in %clk: !seq.clock, in %conda: i1, in %condb: i1, in %val : i32) {
  %true = hw.constant true
  %false = hw.constant false

  %1 = sim.fmt.literal "#1"
  sim.print %1 on %clk if %conda
  %2 = sim.fmt.literal "#2"
  sim.print %2 on %clk if %condb
  %3 = sim.fmt.literal "#3"
  sim.print %3 on %clk if %condb
  %cdis = sim.fmt.literal "--"
  sim.print %cdis on %clk if %false
  %4 = sim.fmt.literal "#4"
  sim.print %4 on %clk if %condb
  %5 = sim.fmt.literal "#5"
  sim.print %5 on %clk if %conda
  %cen = sim.fmt.bin %val {specifierWidth = 32 : i32} : i32
  sim.print %cen on %clk if %true
  %6 = sim.fmt.literal "#6"
  sim.print %6 on %clk if %conda
}

// CHECK-LABEL: @condition_as_val
// CHECK-NEXT:  %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT:  hw.triggered posedge %[[TRG]](%condval) : i1 {
// CHECK-NEXT:  ^bb0(%[[ARG:.*]]: i1):
// CHECK-NEXT:    %[[BIN:.*]] = sim.fmt.bin %[[ARG]] {specifierWidth = 1 : i32} : i1
// CHECK-NEXT:    scf.if %[[ARG]] {
// CHECK-NEXT:      sim.proc.print %[[BIN]]
// CHECK-NEXT:    }
// CHECK-NEXT:  }

hw.module @condition_as_val(in %clk: !seq.clock, in %condval: i1) {
    %bin = sim.fmt.bin %condval {specifierWidth = 1 : i32} : i1
    sim.print %bin on %clk if %condval
}
