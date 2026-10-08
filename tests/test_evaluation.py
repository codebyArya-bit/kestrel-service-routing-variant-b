import unittest
from evaluate import split_by_time, evaluate_predictions

class EvaluationTests(unittest.TestCase):
    def test_boundaries(self):
        rows = [{'created_at_ist': d} for d in ['2025-04-01 00:00','2025-12-31 23:59','2026-01-01 00:00','2026-03-31 23:59','2026-04-01 00:00','2026-06-30 23:59']]
        result = split_by_time(rows)
        self.assertEqual([len(result[k]) for k in ['train','dev','holdout']], [2,2,2])
        self.assertEqual(result, {'train': rows[:2], 'dev': rows[2:4], 'holdout': rows[4:]})
        self.assertTrue(all(r['created_at_ist'] < '2026-04-01' for r in result['train'] + result['dev']))
    def test_invalid_calendar_timestamp(self):
        for timestamp in ['2026-02-99 12:00', '2026-02-31', '2026-04-01 25:00']:
            with self.subTest(timestamp=timestamp):
                with self.assertRaisesRegex(ValueError, 'Invalid created_at_ist timestamp'):
                    split_by_time([{'created_at_ist': timestamp}])

    def test_known_metrics(self):
        result = evaluate_predictions(['Billing','Billing','Repairs','Repairs'], ['Billing','Repairs','Repairs','Repairs'])
        self.assertEqual(result['accuracy'], .75)
        self.assertAlmostEqual(result['macro_f1'], (2/3 + .8)/7)
        self.assertEqual(result['confusion_matrix'][3][1], 1)

class GuardParityTests(unittest.TestCase):
    def test_shared_guard(self):
        from routing.model import nonactionable, route, demo_model
        vague = {'request_text':'Please call back regarding air fryer'}
        self.assertTrue(nonactionable(vague))
        self.assertFalse(nonactionable({'request_text':'invoice missing'}))
        model = demo_model()
        model.config.update(score_threshold=0, margin_threshold=0)
        self.assertTrue(route(model,vague)['review_required'])
