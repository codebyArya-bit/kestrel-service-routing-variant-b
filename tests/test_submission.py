import csv
import tempfile
import unittest
from pathlib import Path
from evaluate import write_predictions

class SubmissionTests(unittest.TestCase):
    def test_contract(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'p.csv'
            write_predictions(['r2','r1'], ['Installations','Consumables'], path)
            with path.open() as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows, [['request_id','team'],['r2','Installations'],['r1','Consumables']])
    def test_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            for ids, teams in [(['x','x'],['Billing','Repairs']), (['x'],['Installs & Demo']), (['x'],[]), ([''],['Billing'])]:
                with self.assertRaises(ValueError):
                    write_predictions(ids, teams, Path(d)/'p.csv')
