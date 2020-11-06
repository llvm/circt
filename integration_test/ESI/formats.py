import lit.Test as Test
import lit.formats

import os
import re
import subprocess
import sys

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
        self.srcdir = os.path.dirname(self.file)
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
            sources = [self.file]
        else:
            sources = [(src if os.path.isabs(src) else os.path.join(
                self.srcdir, src)) for src in self.sources]
        if len(self.cppSources) == 0:
            cppSources = [os.path.join(formatsDir, "..", "driver.cpp")]
        else:
            cppSources = [(src if os.path.isabs(src) else os.path.join(
                self.srcdir, src)) for src in self.cppSources]

        cfg = self.test.config
        cosimInclude = os.path.join(
            cfg.circt_include_dir, "circt", "Dialect", "ESI", "cosim")
        sources.insert(0, os.path.join(cosimInclude, "Cosim_DpiPkg.sv"))
        sources.insert(1, os.path.join(cosimInclude, "Cosim_Endpoint.sv"))
        sources = " ".join(sources + cppSources)
        os.makedirs(self.execdir, exist_ok=True)
        vrun = subprocess.run(
            f"{cfg.verilator_path} --cc --top-module {self.top} --build --exe {sources} {cfg.esi_cosim_path} -LDFLAGS '-Wl,-rpath={cfg.circt_shlib_dir}'".split(),
            capture_output=True,
            text=True,
            cwd=self.execdir)
        output = vrun.stdout + "\n----- STDERR ------\n" + vrun.stderr
        return Test.Result(Test.PASS if vrun.returncode == 0 else Test.FAIL, output)

    def run(self):
        with open(os.path.join(self.execdir, "script.py"), "w") as script:
            cfg = self.test.config
            allDicts = list(self.test.config.__dict__.items()) + \
                list(self.litConfig.__dict__.items())
            vars = dict([(key, value)
                         for (key, value) in allDicts if isinstance(value, str)])
            vars["execdir"] = self.execdir
            vars["srcdir"] = self.srcdir
            vars["srcfile"] = self.file
            vars["rpcSchemaPath"] = os.path.join(
                cfg.circt_include_dir, "circt", "Dialect", "ESI", "cosim",
                "CosimDpi.capnp")
            script.writelines(f"{name} = \"{value}\"\n" for (
                name, value) in vars.items())
            script.write("\n\n")

            script.write(f"import sys\n")
            script.write(
                f"sys.path.append(\"{os.path.dirname(self.file)}\")\n")
            script.writelines(f"import {imp}\n" for imp in self.imports)
            script.write("\n\n")

            script.writelines(f"{x}\n" for x in self.runs)

        timedout = False
        try:
            simStdout = open(os.path.join(self.execdir, "sim_stdout.log"), "w")
            simStderr = open(os.path.join(self.execdir, "sim_stderr.log"), "w")
            simProc = subprocess.Popen([f"./obj_dir/V{self.top}", "--cycles", "-1"],
                                       stdout=simStdout, stderr=simStderr, cwd=self.execdir)

            testStdout = open(os.path.join(
                self.execdir, "test_stdout.log"), "w")
            testStderr = open(os.path.join(
                self.execdir, "test_stderr.log"), "w")
            timeout = None
            if self.litConfig.maxIndividualTestTime > 0:
                timeout = self.litConfig.maxIndividualTestTime
            testProc = subprocess.run([sys.executable, "script.py"],
                                      stdout=testStdout, stderr=testStderr,
                                      timeout=timeout, cwd=self.execdir)
        except TimeoutError:
            timedout = True
        finally:
            if simProc:
                simProc.terminate()
            simStderr.close()
            simStdout.close()
        err, logs = self.readLogs()
        logs += f"---- Test process exit code: {testProc.returncode}\n"
        if timedout:
            result = Test.TIMEOUT
        else:
            result = Test.PASS if testProc.returncode == 0 and not err else Test.FAIL
        return Test.Result(result, logs)

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
