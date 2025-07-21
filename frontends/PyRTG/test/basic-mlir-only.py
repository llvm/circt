# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s

from pyrtg import test, config, Config, report_success, report_failure

# CHECK-LABEL: rtg.target @Singleton : !rtg.dict<>
# CHECK-NEXT: }


@config
class Singleton(Config):
  pass


# CHECK-LABEL: rtg.test @test94_report_test_result
# CHECK-NEXT: rtg.test.success
# CHECK-NEXT: rtg.test.failure "this is a failure message"
# CHECK-NEXT: }


@test(Singleton)
def test94_report_test_result(config):
  report_success()
  report_failure("this is a failure message")
