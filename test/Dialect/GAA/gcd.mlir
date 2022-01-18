"builtin.module"() ({
  "hw.module"() ({
  ^bb0(%write: i32, %writeEnable: i1, %clock: i1, %reset: i1):
    %init = "hw.constant"() {value = 0 : i32} : () -> i32
    %read = "seq.compreg"(%next, %clock, %reset, %init) {name = "r0"} : (i32, i1, i1, i32) -> i32
    %next = "comb.mux"(%writeEnable, %write, %read) : (i1, i32, i32) -> i32
    %writeReady = "hw.constant"() {value = 1 : i1} : () -> i1
    %readReady = "hw.constant"() {value = 1 : i1} : () -> i1
    "hw.output"(%writeReady, %readReady, %read) : (i1, i1, i32) -> ()
  }) {
    argNames = ["write", "writeEnable", "clock", "reset"],
    comment = "",
    parameters = [],
    resultNames = ["writeReady", "readReady", "ready"],
    sym_name = "Reg32",
    function_type = (i32, i1, i1, i1) -> (i1, i1, i32)
  } : () -> ()

  "gaa.circuit"() ({
    "gaa.module.extern"() ({
    ^bb0(%clock: i1, %reset: i1):
      // out-of-method gaa.bind bind the scheduling unrelated IOs.
      "gaa.bind.bare"(%clock) {ref = @clock} : (i1) -> ()
      "gaa.bind.bare"(%reset) {ref = @reset} : (i1) -> ()

      "gaa.bind.value"() {
        sym_name = "read",
        function_type = () -> (i1, i32),
        ready = @readReady,
        return = [@read]
      } : () -> ()

      "gaa.bind.method"() ({
        ^bb0(%writeEnable: i1, %write: i32):
      }) {
        sym_name = "write",
        function_type = (i1, i32) -> (i1),
        enable = @writeEnable,
        argument = [@write],
        ready = @writeReady,
        return = []
      } : () -> ()

    }) {
      sym_name = "GAA_Reg32",
      extModuleName = @Reg32,
      conflict = [[@write, @write]],
      conflictFree = [[@read, @read]],
      sequenceBefore = [[@read, @write]],
      sequenceAfter = [[@write, @read]]
    } : () -> ()

    "gaa.module"() ({
    ^bb0(%clock: i1, %reset: i1):
      "gaa.instance"(%clock, %reset) {moduleName = @GAA_Reg32, sym_name = "x"} : (i1, i1) -> ()
      "gaa.instance"(%clock, %reset) {moduleName = @GAA_Reg32, sym_name = "y"} : (i1, i1) -> ()

      "gaa.rule"() ({
        %ready0, %0 = "gaa.value.call"() {instanceName = @x, functionName = @read} : () -> (i1, i32)
        %ready1, %1 = "gaa.value.call"() {instanceName = @y, functionName = @read} : () -> (i1, i32)

        %writeEnable = "hw.constant"() {value = 1 : i1} : () -> i1
        %ready2 = "gaa.method.call"(%writeEnable, %1) {instanceName = @x, functionName = @write} : (i1, i32) -> (i1)
        %ready3 = "gaa.method.call"(%writeEnable, %0) {instanceName = @y, functionName = @write} : (i1, i32) -> (i1)

        // comb.icmp sgt %0, %1 : i1
        %2 = "comb.icmp"(%0, %1) {predicate = 8 : i64} : (i32, i32) -> i1
        %3 = "hw.constant"() {value = 0 : i32} : () -> i32
        // comb.icmp ne %0, %3 : i1
        %4 = "comb.icmp"(%1, %3) {predicate = 1 : i64} : (i32, i32) -> i1
        %5 = "comb.and"(%2, %4) : (i1, i1) -> i1

        %6 = "comb.and"(%ready0, %ready1, %ready2, %ready3, %5) : (i1, i1, i1, i1, i1) -> i1

        "gaa.rule.return"(%6) : (i1) -> ()
      }) {sym_name = "Swap"} : () -> ()

      "gaa.rule"() ({
        %ready0, %0 = "gaa.value.call"() {instanceName = @x, functionName = @read} : () -> (i1, i32)
        %ready1, %1 = "gaa.value.call"() {instanceName = @y, functionName = @read} : () -> (i1, i32)
        // comb.icmp ule %0, %1 : i1
        %2 = "comb.icmp"(%0, %1) {predicate = 2 : i64} : (i32, i32) -> i1
        %3 = "hw.constant"() {value = 0 : i32} : () -> i32
        // comb.icmp ne %0, %3 : i1
        %4 = "comb.icmp"(%1, %3) {predicate = 0 : i64} : (i32, i32) -> i1
        %5 = "comb.and"(%2, %4) : (i1, i1) -> i1
        %6 = "comb.sub"(%1, %0) : (i32, i32) -> i32
        %writeEnable = "hw.constant"() {value = 1 : i1} : () -> i1
        %ready2 = "gaa.method.call"(%writeEnable, %6) {instanceName = @y, functionName = @write} : (i1, i32) -> (i1)

        %7 = "comb.and"(%ready0, %ready1, %5) : (i1, i1, i1) -> i1

        "gaa.rule.return"(%7) : (i1) -> ()
      }) {sym_name = "Subtract"} : () -> ()

      "gaa.method"() ({
        ^bb0(%enable: i1, %a: i32, %b: i32):
          %0 = "hw.constant"() {value = 0 : i32} : () -> i32
          %ready0, %1 = "gaa.value.call"() {instanceName = @y, functionName = @read} : () -> (i1, i32)
          // comb.icmp eq %0, %1 : i1
          %2 = "comb.icmp"(%0, %1) {predicate = 0 : i64} : (i32, i32) -> i1
          %writeEnable = "hw.constant"() {value = 1 : i1} : () -> i1
          %ready1 = "gaa.method.call"(%writeEnable, %a) {instanceName = @x, functionName = @write} : (i1, i32) -> (i1)
          %ready2 = "gaa.method.call"(%writeEnable, %b) {instanceName = @y, functionName = @write} : (i1, i32) -> (i1)

          %6 = "comb.and"(%ready0, %ready1, %ready2, %2) : (i1, i1, i1, i1) -> i1

          "gaa.method.return"(%6) : (i1) -> ()
      }) {sym_name = "start", function_type = (i1, i32, i32) -> ()} : () -> ()

      "gaa.value"() ({
        %0 = "hw.constant"() {value = 0 : i32} : () -> i32
        %ready0, %1 = "gaa.value.call"() {instanceName = @y, functionName = @read} : () -> (i1, i32)
        // comb.icmp eq %0, %1 : i1
        %2 = "comb.icmp"(%0, %1) {predicate = 0 : i64} : (i32, i32) -> i1
        %3 = "comb.and"(%ready0, %2) : (i1, i1) -> i1
        "gaa.value.return"(%3, %1) : (i1, i32) -> ()
      }) {sym_name = "result", function_type = (i1) -> i32} : () -> ()
    }) {sym_name = "GCD"} : () -> ()
  }) {sym_name = "GCD"} : () -> ()
}) : () -> ()