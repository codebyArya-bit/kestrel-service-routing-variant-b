"""Local employee routing screen. Explicit model paths must be trusted pickles."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from routing.model import demo_model, load_model, route

ENUMS = {
    'channel': ('chat', 'email', 'ivr', 'web', 'whatsapp'),
    'product_family': ('Air Fryer', 'Ceiling Fan', 'Induction Cooktop', 'Mixer Grinder', 'Robot Vacuum', 'Room Heater', 'Water Purifier', 'appliance'),
    'warranty_status': ('in_warranty', 'out_of_warranty', 'shield', 'unknown'),
}


def select_model(path):
    model = load_model(Path(path)) if path is not None else demo_model()
    model.config['mode'] = 'private' if path is not None else 'demo'
    return model


def create_server(model, host='127.0.0.1', port=8766):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Do not log requests or intake text.

        def respond(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/':
                self.respond(404, {'error': 'Page not found.'})
                return
            body = (Path(__file__).parent / 'web' / 'index.html').read_bytes()
            label = 'Synthetic demo — synthetic training only' if model.config['mode'] == 'demo' else 'Private model — employee decision support'
            body = body.replace(b'{{MODE}}', label.encode())
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != '/api/route':
                self.respond(404, {'error': 'Endpoint not found.'})
                return
            try:
                length = int(self.headers.get('Content-Length', ''))
                if length < 0:
                    raise ValueError
            except ValueError:
                self.respond(400, {'error': 'A valid Content-Length is required.'})
                return
            if length > 65536:
                self.respond(413, {'error': 'Request body exceeds 65536 bytes.'})
                return
            self.connection.settimeout(5)
            try:
                record = json.loads(self.rfile.read(length))
                if not isinstance(record, dict):
                    raise ValueError('Send a JSON object.')
                text = record.get('request_text')
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('Request text is required and must be nonempty text.')
                if len(text) > 10000:
                    raise ValueError('Request text exceeds 10000 characters.')
                for key, allowed in ENUMS.items():
                    if key in record and record[key] not in allowed:
                        raise ValueError(f'Unknown {key}; choose a listed value.')
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                self.respond(400, {'error': 'Send valid UTF-8 JSON.'})
                return
            except ValueError as exc:
                self.respond(400, {'error': str(exc)})
                return
            except (TimeoutError, OSError):
                self.respond(400, {'error': 'Request body could not be read.'})
                return
            try:
                result = route(model, record)
            except Exception:
                self.respond(500, {'error': 'Routing failed. Please retry or request human review.'})
                return
            self.respond(200, result)
    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Trusted private model pickle; omitted uses synthetic demo')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    try:
        model = select_model(args.model)
        server = create_server(model, port=args.port)
    except (OSError, ValueError, EOFError) as exc:
        parser.exit(2, f'Cannot start routing service: {exc}\n')
    print(f'Kestrel {model.config["mode"]} mode: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
