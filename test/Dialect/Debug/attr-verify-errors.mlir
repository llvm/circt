// RUN: circt-opt %s --verify-diagnostics --split-input-file

// expected-error @+1 {{'dbg.moduleinfo' must be a dictionary}}
hw.module @NotDictionary() attributes {dbg.moduleinfo = "Alu"} {}

// -----

// expected-error @+1 {{'dbg.moduleinfo' name must be a string}}
hw.module @NameNotString() attributes {dbg.moduleinfo = {name = 1 : i32}} {}

// -----

// expected-error @+1 {{'dbg.moduleinfo' ext must be a dictionary}}
hw.module @ExtNotDictionary() attributes {dbg.moduleinfo = {name = "Alu", ext = "x"}} {}

// -----

// expected-error @+1 {{ext key 'Chisel' is not a lowercase language name}}
hw.module @ExtBadLanguage() attributes {dbg.moduleinfo = {name = "Alu", ext = {Chisel = "{}"}}} {}

// -----

// expected-error @+1 {{'dbg.merged_from' must be a string}}
hw.module @MergedFromNotString() attributes {dbg.merged_from = 1 : i32} {}

// -----

// expected-error @+1 {{'dbg.lowered' must be a unit attribute}}
hw.module @LoweredNotUnit() attributes {dbg.lowered = true} {}

// -----

// expected-error @+1 {{'dbg.artificial' must be a unit attribute}}
hw.module @ArtificialNotUnit() attributes {dbg.artificial = "x"} {}

// -----

// expected-error @+1 {{'dbg.memory' must be a unit attribute}}
hw.module @MemoryNotUnit() attributes {dbg.memory = 1 : i32} {}

// -----

// expected-error @+1 {{'dbg.module_locs' must map module names to locations}}
hw.module @ModuleLocsNotDictionary() attributes {dbg.module_locs = "x"} {}

// -----

// expected-error @+1 {{'dbg.module_locs' must map module names to locations}}
hw.module @ModuleLocsNotLocation() attributes {dbg.module_locs = {A = "x"}} {}

// -----

// expected-error @+1 {{has unknown attribute 'dbg.lowred'}}
hw.module @Unknown() attributes {dbg.lowred} {}
