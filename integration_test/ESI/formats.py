import lit.Test as Test
import lit.formats

import os
import importlib
from pprint import pprint
import subprocess
import sys
import re
import traceback
import multiprocessing

formatsDir = os.path.dirname(__file__)


class CosimTest(lit.formats.FileBasedTest):
    def execute(self, test, litConfig):
        if 'esi-cosim' not in test.config.available_features:
            return Test.Result(Test.UNSUPPORTED, "ESI-Cosim feature not present")
        if 'verilator' not in test.config.available_features:
            return Test.Result(Test.UNSUPPORTED, "Verilator not present")
        testRun = CosimTestRunner(test)
        parsed = testRun.parse()
        if parsed.code != Test.PASS:
            return parsed
        compiled = testRun.compile()
        if compiled.code != Test.PASS:
            return compiled

        return testRun.run()


class CosimTestRunner:
    def __init__(self, test):
        self.test = test
        self.file = test.getSourcePath()
        self.execdir = test.getExecPath()
        self.runs = list()
        self.sources = list()
        self.cppSources = list()
        self.imports = list()
        self.top = "main"

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
        if not os.path.exists(self.execdir):
            os.mkdir(self.execdir)
        vrun = subprocess.run(
            f"{cfg.verilator_path} --cc --top-module {self.top} --build --exe {sources} {cfg.esi_cosim_path} -LDFLAGS '-Wl,-rpath={cfg.circt_shlib_dir}'".split(),
            capture_output=True,
            text=True,
            cwd=self.execdir)
        output = vrun.stdout + "\n----- STDERR ------\n" + vrun.stderr
        return Test.Result(Test.PASS if vrun.returncode == 0 else Test.FAIL, output)

    def run(self):
        origdir = os.getcwd()
        os.chdir(self.execdir)
        try:
            testStdout = open("test_stdout.log", "w")
            testStderr = open("test_stderr.log", "w")

            simStdout = open("sim_stdout.log", "w")
            simStderr = open("sim_stderr.log", "w")
            simProc = subprocess.Popen([f"./obj_dir/V{self.top}", "--cycles", "-1"],
                                       stdout=simStdout, stderr=simStderr)

            origStdout = sys.stdout
            origStderr = sys.stderr
            sys.stdout = testStdout
            sys.stderr = testStderr
            try:
                result = self.runPyTest()
            except Exception:
                result = False
                traceback.print_exc()
            finally:
                sys.stdout = origStdout
                sys.stderr = origStderr
                simProc.kill()
                testStderr.close()
                testStdout.close()
                simStderr.close()
                simStdout.close()
                msg = self.readLogs()
        finally:
            os.chdir(origdir)
        return Test.Result(Test.PASS if result else Test.FAIL, msg)

    def runPyTest(self):
        cfg = self.test.config
        rpcSchemaPath = os.path.join(
            cfg.circt_include_dir, "circt", "Dialect", "ESI", "cosim", "CosimDpi.capnp")
        for imp in self.imports:
            exec(f"import {imp}")
        for run in self.runs:
            exec(run)
        return True

    def readLogs(self):
        ret = "----- Simulation stdout -----\n"
        with open("sim_stdout.log") as f:
            ret += f.read()

        with open("sim_stderr.log") as f:
            stderr = f.read()
            if stderr != "":
                ret += "\n----- Simulation stderr -----\n"
                ret += stderr

        ret += "\n----- Test stdout -----\n"
        with open("test_stdout.log") as f:
            ret += f.read()

        with open("test_stderr.log") as f:
            stderr = f.read()
            if stderr != "":
                ret += "\n----- Test stderr -----\n"
                ret += stderr

        return ret
