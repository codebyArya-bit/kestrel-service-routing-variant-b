import http.client
import json
import threading
import unittest
from pathlib import Path
from routing.model import demo_model, route
from serve import create_server, select_model


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = demo_model()
        cls.server = create_server(cls.model, port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, body, headers=None, method='POST', path='/api/route'):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        conn.request(method, path, body, headers or {'Content-Type': 'application/json'})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def test_success_matches_batch(self):
        record = {'request_text': 'Please install my new purifier', 'channel': 'web', 'product_family': 'appliance', 'warranty_status': 'unknown'}
        status, body = self.request(json.dumps(record))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), route(self.model, record))

    def test_invalid_inputs(self):
        for body in ['{', '[]', '{}', '{"request_text":"  "}', json.dumps({'request_text': 'x'*10001}), json.dumps({'request_text': 'repair', 'channel': 'bogus'}), json.dumps({'request_text': 'repair', 'product_family': 'bogus'}), json.dumps({'request_text': 'repair', 'warranty_status': 'bogus'}), '{"request_text":5}']:
            with self.subTest(body=body[:30]):
                status, data = self.request(body)
                self.assertEqual(status, 400)
                self.assertIsInstance(json.loads(data)['error'], str)

    def test_aliases_and_precise_guard_reason(self):
        for text, canonical, alias in [('install purifier technician setup', 'Installations', 'Installs & Demo'), ('filter cartridge consumable supplies', 'Consumables', 'Filters & Consumables')]:
            result = route(self.model, {'request_text': text})
            self.assertEqual(result['team'], canonical)
            self.assertEqual(result['display_team'], alias)
        result = route(self.model, {'request_text': 'hello'})
        self.assertTrue(result['review_required'])
        self.assertTrue(any('lacks recognized actionable' in r for r in result['reasons']))
        self.assertNotIn('Human review required: model support is low or competing teams are close.', result['reasons'])

    def test_body_limit(self):
        self.assertEqual(self.request(' '*65537)[0], 413)

    def test_deeply_nested_json(self):
        status, data = self.request('[' * 30000 + '0' + ']' * 30000)
        self.assertEqual(status, 400)
        self.assertIsInstance(json.loads(data)['error'], str)

    def test_bad_length(self):
        self.assertEqual(self.request('', {'Content-Length':'invalid'})[0], 400)

    def test_screen(self):
        status, body = self.request(None, method='GET', path='/')
        self.assertEqual(status, 200)
        self.assertIn(b'textContent', body)
        self.assertIn(b'Synthetic demo', body)

    def test_modes_and_missing_artifact(self):
        self.assertEqual(select_model(None).config['mode'], 'demo')
        with self.assertRaises(FileNotFoundError):
            select_model(Path('/missing/model.pkl'))

if __name__ == '__main__':
    unittest.main()
