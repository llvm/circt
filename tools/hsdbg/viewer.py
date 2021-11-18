import logging
from flask import Flask, render_template, send_file, jsonify
from threading import Thread


class HSDbgImageServer:
  # An image server which can be used to display the current state of the HSDbg
  # graph.

  def __init__(self, hsdbg, port):
    self.hsdbg = hsdbg
    self.port = port
    self.server = None

  def finish(self):
    return

  def start(self):
    """ Starts the image server in a separate thread."""

    def runServer():
      # A simple flask application which serves index.html to continuously update
      # the svg file.
      log = logging.getLogger('werkzeug')
      log.setLevel(logging.ERROR)
      app = Flask(__name__)
      app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

      @app.route("/")
      def index():
        return render_template("index.html")

      @app.route("/cycle")
      def cycle():
        return jsonify(cycle=self.hsdbg.currentCycle())

      # Serve the svg file on any path request.
      @app.route("/<path:path>")
      def serveFile(path):
        return send_file(self.hsdbg.workingImagePath, mimetype="image/svg+xml")

      app.run(host="localhost", port=self.port)

    self.server = Thread(target=lambda: runServer())
    self.server.start()
