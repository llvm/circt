import lit.Test as Test
import lit.formats

import os
import re
import signal
import socket
import subprocess
import sys
import time


class CosimTest(lit.formats.FileBasedTest):
    """A lit test format (adapter) for running cosimulation-based tests."""

    def execute(self, test, litConfig):
        # All cosim tests require esi-cosim to have been built.
        if 'esi-cosim' not in test.config.available_features:
            return Test.Result(Test.UNSUPPORTED, "ESI-Cosim feature not present")
        # This format is specialized to Verilator at the moment.
        if 'verilator' not in test.config.available_features:
            return Test.Result(Test.UNSUPPORTED, "Verilator not present")

        # Parse, compile, and run the test.
        testRun = CosimTestRunner(test, litConfig)
        parsed = testRun.parse()
        if parsed.code != Test.PASS:
            return parsed
        compiled = testRun.compile()
        if compiled.code != Test.PASS:
            return compiled
        return testRun.run()


class CosimTestRunner:
    """The main class responsible for running a cosim test. We use a separate
    class to allow for per-test mutable state variables."""

    def __init__(self, test, litConfig):
        self.test = test
        self.litConfig = litConfig
        self.file = test.getSourcePath()
        self.srcdir = os.path.dirname(self.file)
        self.execdir = test.getExecPath()
        self.runs = list()
        self.sources = list()
        self.cppSources = list()
        self.top = "main"

    def parse(self):
        """Parse a test file. Look for comments we recognize anywhere in the
        file."""
        fileReader = open(self.file, "r")
        for line in fileReader:
            # TOP is the top module.
            if m := re.match(r"^//\s*TOP:(.*)$", line):
                self.top = m.group(1)
            # FILES are the additional RTL files (if any). If specified, must
            # include the current file. These files are either absolute or
            # relative to the current file.
            if m := re.match(r"^//\s*FILES:(.*)$", line):
                self.sources.extend(m.group(1).split())
            # C++ driver files to feed to Verilator. Defaults to the simple
            # driver.cpp. Same path rules as FILES.
            if m := re.match(r"^//\s*CPP:(.*)$", line):
                self.cppSources.extend(m.group(1).split())
            # Run this Python line.
            if m := re.match(r"^//\s*PY:(.*)$", line):
                self.runs.append(m.group(1).strip())
        fileReader.close()
        return Test.Result(Test.PASS, "")

    def compile(self):
        """Compile the simulation with Verilator. Let Verilator do the whole
        thing and produce a binary which 'just works'. This is sufficient for
        simple-ish C++ drivers."""

        # Assemble a list of sources (RTL and C++), applying the defaults and
        # path rules.
        if len(self.sources) == 0:
            sources = [self.file]
        else:
            sources = [(src if os.path.isabs(src) else os.path.join(
                self.srcdir, src)) for src in self.sources]
        if len(self.cppSources) == 0:
            cppSources = [os.path.join(
                os.path.dirname(__file__), "..", "driver.cpp")]
        else:
            cppSources = [(src if os.path.isabs(src) else os.path.join(
                self.srcdir, src)) for src in self.cppSources]

        # Include the cosim DPI SystemVerilog files.
        cfg = self.test.config
        cosimInclude = os.path.join(
            cfg.circt_include_dir, "circt", "Dialect", "ESI", "cosim")
        sources.insert(0, os.path.join(cosimInclude, "Cosim_DpiPkg.sv"))
        sources.insert(1, os.path.join(cosimInclude, "Cosim_Endpoint.sv"))

        # Format the list of sources.
        sources = " ".join(sources + cppSources)
        os.makedirs(self.execdir, exist_ok=True)

        # Run verilator to produce an executable. Requires a working Verilator
        # install in the PATH.
        vrun = subprocess.run(
            f"{cfg.verilator_path} --cc --top-module {self.top} --build --exe {sources} {cfg.esi_cosim_path} -LDFLAGS '-Wl,-rpath={cfg.circt_shlib_dir}'".split(),
            capture_output=True,
            text=True,
            cwd=self.execdir)
        output = vrun.stdout + "\n----- STDERR ------\n" + vrun.stderr
        return Test.Result(Test.PASS if vrun.returncode == 0 else Test.FAIL, output)

    def run(self):
        """Run the test by creating a Python script, starting the simulation,
        running the Python script, then stopping the simulation.

        Not perfect since we don't know when the cosim RPC server in the
        simulation has started accepting connections."""

        # Find available port.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()

        with open(os.path.join(self.execdir, "script.py"), "w") as script:
            # Include a bunch of config variables at the beginning of the
            # script for use by the test code.
            cfg = self.test.config
            allDicts = list(self.test.config.__dict__.items()) + \
                list(self.litConfig.__dict__.items())
            vars = dict([(key, value)
                         for (key, value) in allDicts if isinstance(value, str)])
            vars["execdir"] = self.execdir
            vars["srcdir"] = self.srcdir
            vars["srcfile"] = self.file
            # 'rpcSchemaPath' points to the CapnProto schema for RPC and is the
            # one that nearly all scripts are going to need.
            vars["rpcschemapath"] = os.path.join(
                cfg.circt_include_dir, "circt", "Dialect", "ESI", "cosim",
                "CosimDpi.capnp")
            script.writelines(f"{name} = \"{value}\"\n" for (
                name, value) in vars.items())
            script.write("\n\n")

            # Add the test files directory and this files directory to the
            # pythonpath.
            script.write(f"import os\n")
            script.write(f"import sys\n")
            script.write(
                f"sys.path.append(\"{os.path.dirname(self.file)}\")\n")
            script.write(
                f"sys.path.append(\"{os.path.dirname(__file__)}\")\n")
            script.write("\n\n")
            script.write(
                "simhostport = f'{os.uname()[1]}:" + str(port) + "'\n")

            # Run the lines specified in the test file.
            script.writelines(
                f"{x}\n" for x in self.runs)

        timedout = False
        try:
            # Run the simulation.
            simStdout = open(os.path.join(self.execdir, "sim_stdout.log"), "w")
            simStderr = open(os.path.join(self.execdir, "sim_stderr.log"), "w")
            simEnv = os.environ.copy()
            simEnv["COSIM_PORT"] = str(port)
            simProc = subprocess.Popen(
                [f"./obj_dir/V{self.top}", "--cycles", "-1"],
                stdout=simStdout, stderr=simStderr, cwd=self.execdir,
                env=simEnv)
            # Wait a set amount of time for the simulation to start accepting
            # RPC connections.
            # TODO: Check if the server is up by polling.
            time.sleep(0.05)

            # Run the test script.
            testStdout = open(os.path.join(
                self.execdir, "test_stdout.log"), "w")
            testStderr = open(os.path.join(
                self.execdir, "test_stderr.log"), "w")
            timeout = None
            if self.litConfig.maxIndividualTestTime > 0:
                timeout = self.litConfig.maxIndividualTestTime
            testProc = subprocess.run([sys.executable, "-u", "script.py"],
                                      stdout=testStdout, stderr=testStderr,
                                      timeout=timeout, cwd=self.execdir)
        except subprocess.TimeoutExpired:
            timedout = True
        finally:
            # Make sure to stop the simulation no matter what.
            if simProc:
                simProc.send_signal(signal.SIGINT)
                # Allow the simulation time to flush its outputs.
                try:
                    simProc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    simProc.kill()
            simStderr.close()
            simStdout.close()
            testStdout.close()
            testStderr.close()

        # Read the output log files and return the proper result.
        err, logs = self.readLogs()
        if timedout:
            result = Test.TIMEOUT
        else:
            logs += f"---- Test process exit code: {testProc.returncode}\n"
            result = Test.PASS if testProc.returncode == 0 and not err else Test.FAIL
        return Test.Result(result, logs)

    def readLogs(self):
        """Read the log files from the simulation and the test script. Only
        add the stderr logs if they contain something. Also return a flag
        indicating that one of the stderr logs has content."""

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
