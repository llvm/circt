# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s --check-prefix=MLIR
# RUN: %rtgtool% %s --seed=0 --output-format=elaborated | FileCheck %s --check-prefix=ELABORATED
# RUN: %rtgtool% %s --seed=0 -o %t --output-format=asm | FileCheck %s --input-file=%t --check-prefix=ASM

from pyrtg import test

# MLIR: rtg.test @test0
# MLIR-NEXT: }

# ELABORATED: rtg.test @test0
# ELABORATED-NEXT: }

# ASM: Begin of test0
# ASM: End of test0


@test
def test0():
  pass
