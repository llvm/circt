#include "esi/Accelerator.h"
#include "esi/CLI.h"
#include "esi/SimplerManifest.h"

#include "esi/backends/RpcServer.h"

using namespace esi;

constexpr std::string_view testManifest = R"json(
{
  "service_decls": [
    {
      "type": "TelemetryService",
      "id": {"name": "telemetry_service"}
    }
  ],
  "ports": [
    {
      "name": "foo_args",
      "cosim_channel_name": "foo_args_chan",
      "direction": "to"
    }
  ]
}
)json";

int runServer(CliParser &cli);

int main(int argc, const char *argv[]) {
  CliParser cli("simpler_manifest_test");

  CLI::App *server = cli.add_subcommand(
      "server", "Run the simpler manifest test server (backend is ignored, "
                "conn string is the port)");

  if (int rc = cli.esiParse(argc, argv))
    return rc;
  if (!cli.get_help_ptr()->empty())
    return 0;

  if (*server)
    return runServer(cli);

  Context &ctxt = cli.getContext();
  AcceleratorConnection *acc = cli.connect();
  SimplerManifest manifest(ctxt, std::string(testManifest));
  Accelerator *design = manifest.buildAccelerator(*acc);

  const BundlePort &foo_arg_bundle = design->getPorts().at(AppID("foo_args"));
  WriteChannelPort &foo_arg_port = foo_arg_bundle.getRawWrite("data");
  foo_arg_port.connect();
  foo_arg_port.write(MessageData::from("Hello, world!"));

  return 0;
}

int runServer(CliParser &cli) {
  Context &ctxt = cli.getContext();

  int port = std::stoi(cli.getConnectionString());
  cosim::RpcServer server(ctxt);
  server.run(port);

  ReadChannelPort &foo_args =
      server.registerReadPort("foo_args_chan", "string");
  foo_args.connect();
  while (true) {
    MessageData msg;
    foo_args.read(msg);
    std::string receivedStr(reinterpret_cast<const char *>(msg.getBytes()),
                            msg.getSize());
    std::cout << "Received message: " << receivedStr << "\n";
  }
  return 0;
}
