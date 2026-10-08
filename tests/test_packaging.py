import tempfile
import unittest
from pathlib import Path
from scripts.package import bundle, public_paths, scan_public, sha


class PackagingTests(unittest.TestCase):
    def test_explicit_allowlist_excludes_private_and_plans(self):
        paths = [str(p) for p in public_paths(Path('/repo'))]
        self.assertFalse(any('private-' in p or 'superpowers' in p for p in paths))

    def test_scan_rejects_real_identifier(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'README.md'
            p.write_text('leaked REQUEST-PRIVATE')
            with self.assertRaises(ValueError):
                scan_public([p], {b'REQUEST-PRIVATE'})
            p.write_text('generic synthetic demo')
            scan_public([p], {b'REQUEST-PRIVATE'})

    def test_zip_reproducible_and_manifest(self):
        import json
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'file.txt'
            p.write_text('same bytes')
            a, b = Path(d)/'a.zip', Path(d)/'b.zip'
            bundle(a, {'file.txt': p})
            bundle(b, {'file.txt': p})
            self.assertEqual(sha(a), sha(b))
            with zipfile.ZipFile(a) as z:
                self.assertEqual(json.loads(z.read('manifest-sha256.json')), {'file.txt': sha(p)})

    def test_rejects_mutated_frozen_artifacts(self):
        import json
        from scripts.package import check_frozen_artifacts
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            final = root / 'private-output/final'
            final.mkdir(parents=True)
            model, freeze = final/'model.pkl', final/'freeze.json'
            model.write_bytes(b'synthetic trained artifact')
            freeze.write_text('{"synthetic": true}')
            baseline_model, baseline_freeze = model.read_bytes(), freeze.read_bytes()
            expected_model, expected_freeze = sha(model), sha(freeze)
            evaluation = {'model_sha256': expected_model, 'freeze_sha256': expected_freeze}
            (final/'evaluation.json').write_text(json.dumps(evaluation))
            (final/'freeze.sha256').write_text(expected_freeze + '\n')
            self.assertTrue(check_frozen_artifacts(root)['evaluation_consistency'])
            model.write_bytes(b'mutated synthetic artifact')
            with self.assertRaisesRegex(ValueError, 'Frozen model hash mismatch'):
                check_frozen_artifacts(root)
            model.write_bytes(baseline_model)
            freeze.write_text('{"synthetic": false}')
            with self.assertRaisesRegex(ValueError, 'Frozen manifest hash mismatch'):
                check_frozen_artifacts(root)
            # Even an updated sidecar cannot override frozen evaluation evidence.
            (final/'freeze.sha256').write_text(sha(freeze))
            with self.assertRaisesRegex(ValueError, 'Frozen manifest hash mismatch'):
                check_frozen_artifacts(root)
            freeze.write_bytes(baseline_freeze)
            (final/'freeze.sha256').write_text('incorrect sidecar')
            with self.assertRaisesRegex(ValueError, 'Frozen manifest hash mismatch'):
                check_frozen_artifacts(root)
