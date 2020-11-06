import lit.Test as Test
import lit.formats

import os
import importlib
from pprint import pprint
import subprocess
import re
import signal
import sys
import traceback
import time

formatsDir = os.path.dirname(__file__)


class CosimTest(lit.formats.FileBasedTest):
    def execute(self, test, litConfig):
        if 'esi-cosim' not in test.config.available_features:
            return Test.Result(Test.UNSUPPORTED, "ESI-Cosim feature not present")
        if 'verilator' not in test.config.available_features:
            return Test.Result(Test.UNSUPPORTED, "Verilator not present")
        testRun = CosimTestRunner(test, litConfig)
        parsed = testRun.parse()
        if parsed.code != Test.PASS:
            return parsed
        compiled = testRun.compile()
        if compiled.code != Test.PASS:
            return compiled

        return testRun.run()


class CosimTestRunner:
    def __init__(self, test, litConfig):
        self.test = test
        self.litConfig = litConfig
        self.file = test.getSourcePath()
        self.execdir = test.getExecPath()
        self.runs = list()
        self.sources = list()
        self.cppSources = list()
        self.imports = list()
        self.top = "main"
        if litConfig.maxIndividualTestTime > 0:
            self.deadline = time.time() + litConfig.maxIndividualTestTime
        else:
            self.deadline = None

    def parse(self):
        fileReader = open(self.file, "r")
        for line in fileReader:
            if m := re.match(r"^//\s*TOP:(.*)$", line):
                self.top = m.group(1)
            if m := re.match(r"^//\s*FILES:(.*)$", line):
                self.sources.extend(m.group(1).split())
            if m := re.match(r"^//\s*CPP:(.*)$", line):
                self.cppSources.extend(m.group(1).split())
            if m := re.match(r"^//\s*RUN:(.*)$", line):
                self.runs.append(m.group(1).strip())
            if m := re.match(r"^//\s*IMPORT:(.*)$", line):
                self.imports.append(m.group(1))
        fileReader.close()
        return Test.Result(Test.PASS, "")

    def compile(self):
        if len(self.sources) == 0:
            self.sources.append(self.file)
        if len(self.cppSources) == 0:
            self.cppSources.append(os.path.join(
                formatsDir, "..", "driver.cpp"))

        cfg = self.test.config
        cosimInclude = os.path.join(
            cfg.circt_include_dir, "circt", "Dialect", "ESI", "cosim")
        self.sources.insert(0, os.path.join(cosimInclude, "Cosim_DpiPkg.sv"))
        self.sources.insert(1, os.path.join(cosimInclude, "Cosim_Endpoint.sv"))
        sources = " ".join(self.sources + self.cppSources)
        os.makedirs(self.execdir, exist_ok=True)
        vrun = subprocess.run(
            f"{cfg.verilator_path} --cc --top-module {self.top} --build --exe {sources} {cfg.esi_cosim_path} -LDFLAGS '-Wl,-rpath={cfg.circt_shlib_dir}'".split(),
            capture_output=True,
            text=True,
            cwd=self.execdir)
        output = vrun.stdout + "\n----- STDERR ------\n" + vrun.stderr
        return Test.Result(Test.PASS if vrun.returncode == 0 else Test.FAIL, output)

    def run(self):
        pid = os.fork()
        if pid == 0:
            # Does not return. Runs test and exits.
            signal.signal(signal.SIGINT, signal.default_int_handler)
            self.runTest()

        rc = waitpid_deadline(pid, self.deadline)
        if rc == False:
            os.kill(pid, signal.SIGINT)
            os.kill(pid, signal.SIGINT)
            if waitpid_deadline(pid, time.time() + 5.0) == False:
                print(f"Force kill {pid}")
                os.kill(pid, signal.SIGKILL)
            (_, msg) = self.readLogs()
            return Test.Result(Test.TIMEOUT, msg)

        procrc = rc[1]
        (err, msg) = self.readLogs()
        passed = not err and procrc == 0
        return Test.Result(Test.PASS if passed else Test.FAIL, msg)

    def runTest(self):
        os.chdir(self.execdir)
        try:
            sys.stdout = open("test_stdout.log", "w")
            sys.stderr = open("test_stderr.log", "w")

            simStdout = open("sim_stdout.log", "w")
            simStderr = open("sim_stderr.log", "w")
            simProc = subprocess.Popen([f"./obj_dir/V{self.top}", "--cycles", "-1"],
                                       stdout=simStdout, stderr=simStderr)
            result = False
            try:
                cfg = self.test.config
                rpcSchemaPath = os.path.join(
                    cfg.circt_include_dir, "circt", "Dialect", "ESI", "cosim", "CosimDpi.capnp")
                for imp in self.imports:
                    exec(f"import {imp}")
                for run in self.runs:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    exec(run)
                result = True
            except KeyboardInterrupt:
                pass
            except Exception:
                traceback.print_exc()
            finally:
                simProc.kill()
                simStderr.close()
                simStdout.close()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0 if result else 1)

    def readLogs(self):
        foundErr = False
        ret = "----- Simulation stdout -----\n"
        with open(os.path.join(self.execdir, "sim_stdout.log")) as f:
            ret += f.read()

        with open(os.path.join(self.execdir, "sim_stderr.log")) as f:
            stderr = f.read()
            if stderr != "":
                ret += "\n----- Simulation stderr -----\n"
                ret += stderr
                foundErr = True

        ret += "\n----- Test stdout -----\n"
        with open(os.path.join(self.execdir, "test_stdout.log")) as f:
            ret += f.read()

        with open(os.path.join(self.execdir, "test_stderr.log")) as f:
            stderr = f.read()
            if stderr != "":
                ret += "\n----- Test stderr -----\n"
                ret += stderr
                foundErr = True

        return (foundErr, ret)


def waitpid_deadline(pid, deadline):
    if deadline == None:
        (waitpid, rc) = os.waitpid(pid, 0)
        return (True, rc)

    while time.time() < deadline:
        (waitpid, rc) = os.waitpid(pid, os.WNOHANG)
        if waitpid != 0:
            return (True, rc)
        time.sleep(0.01)
    return False
