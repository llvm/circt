// RUN: circt-test -l -f 'Alpha*' %s | FileCheck %s --check-prefix=INC-NAME
// RUN: circt-test -l -x '*Gamma' %s | FileCheck %s --check-prefix=EXC-NAME
// RUN: circt-test -l -f 'engine=*' %s | FileCheck %s --check-prefix=INC-EXIST
// RUN: circt-test -l -f 'engine=bmc' %s | FileCheck %s --check-prefix=INC-VAL
// RUN: circt-test -l -f '*Alpha*' -x '*Gamma*' %s | FileCheck %s --check-prefix=COMBINED
// RUN: circt-test -l -f '*Alpha*' -f '*Delta*' %s | FileCheck %s --check-prefix=MULTI-INC
// RUN: circt-test -l -x 'depth=*' %s | FileCheck %s --check-prefix=EXC-EXIST
// RUN: circt-test -l -f 'depth=5' %s | FileCheck %s --check-prefix=INC-INT
// RUN: circt-test -l --include '*Alpha*' --exclude '*Gamma*' %s | FileCheck %s --check-prefix=COMBINED
// RUN: circt-test -l -f 'config.backend=vltr' %s | FileCheck %s --check-prefix=NESTED-VAL
// RUN: circt-test -l -f 'config.backend=*' %s | FileCheck %s --check-prefix=NESTED-EXIST
// RUN: circt-test -l -x 'config.backend=*' %s | FileCheck %s --check-prefix=NESTED-EXC
// RUN: circt-test -l -f 'Alph?' %s | FileCheck %s --check-prefix=GLOB-QMARK
// RUN: circt-test -l -f '[AZ]*' %s | FileCheck %s --check-prefix=GLOB-CLASS

// INC-NAME: Alpha formal
// INC-NAME: AlphaGamma formal
// INC-NAME-NOT: Beta
// INC-NAME-NOT: Delta

// EXC-NAME: Alpha formal {
// EXC-NAME: Beta formal
// EXC-NAME-NOT: AlphaGamma
// EXC-NAME: Delta simulation

// INC-EXIST: Alpha formal {
// INC-EXIST: Beta formal
// INC-EXIST-NOT: AlphaGamma
// INC-EXIST: Delta simulation

// INC-VAL: Alpha formal {
// INC-VAL-NOT: Beta
// INC-VAL-NOT: AlphaGamma
// INC-VAL: Delta simulation

// COMBINED: Alpha formal {
// COMBINED-NOT: AlphaGamma
// COMBINED-NOT: Beta
// COMBINED-NOT: Delta

// MULTI-INC: Alpha formal {
// MULTI-INC: AlphaGamma formal
// MULTI-INC-NOT: Beta
// MULTI-INC: Delta simulation

// EXC-EXIST-NOT: depth =
// EXC-EXIST: Beta formal
// EXC-EXIST: AlphaGamma formal
// EXC-EXIST: Delta simulation

// INC-INT: Alpha formal {
// INC-INT-NOT: Beta
// INC-INT-NOT: AlphaGamma
// INC-INT-NOT: Delta

// NESTED-VAL-NOT: Alpha
// NESTED-VAL-NOT: Beta
// NESTED-VAL-NOT: AlphaGamma
// NESTED-VAL-NOT: Delta
// NESTED-VAL: Epsilon formal
// NESTED-VAL-NOT: Zeta

// NESTED-EXIST-NOT: Alpha
// NESTED-EXIST-NOT: Beta
// NESTED-EXIST-NOT: AlphaGamma
// NESTED-EXIST-NOT: Delta
// NESTED-EXIST: Epsilon formal
// NESTED-EXIST: Zeta formal

// NESTED-EXC: Alpha formal
// NESTED-EXC: Beta formal
// NESTED-EXC: AlphaGamma formal
// NESTED-EXC: Delta simulation
// NESTED-EXC-NOT: Epsilon
// NESTED-EXC-NOT: Zeta

// GLOB-QMARK: Alpha formal
// GLOB-QMARK-NOT: AlphaGamma
// GLOB-QMARK-NOT: Beta
// GLOB-QMARK-NOT: Delta

// GLOB-CLASS: Alpha formal
// GLOB-CLASS: AlphaGamma formal
// GLOB-CLASS-NOT: Beta
// GLOB-CLASS-NOT: Delta
// GLOB-CLASS-NOT: Epsilon
// GLOB-CLASS: Zeta formal

verif.formal @Alpha {engine = "bmc", depth = 5 : i64} {}
verif.formal @Beta {engine = "induction"} {}
verif.formal @AlphaGamma {} {}
verif.simulation @Delta {engine = "bmc"} {
^bb0(%clock: !seq.clock, %init: i1):
  %0 = hw.constant true
  verif.yield %0, %0 : i1, i1
}
verif.formal @Epsilon {config = {backend = "vltr", depth = 7 : i64}} {}
verif.formal @Zeta {config = {backend = "icrs", depth = 3 : i64}} {}
