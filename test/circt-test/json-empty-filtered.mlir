// RUN: circt-test -l --json -x 'MyTest' %s 2>/dev/null | FileCheck %s

// CHECK: []

verif.formal @MyTest {} {}
