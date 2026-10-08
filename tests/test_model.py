import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from routing.data import TEAMS
from routing.model import fit_model, route, save_model, load_model

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Path('demo/synthetic.csv').open() as f:
            rows=list(csv.DictReader(f))
        cls.model=fit_model(rows,[r['team'] for r in rows],{})

    def test_excludes_outcomes_and_identifiers(self):
        row={'request_text':'Please install the purifier','channel':'phone'}
        polluted=dict(row,final_team='Billing',team_label='Billing',transfers=50,request_id='Billing',source='Billing',created_at_ist='Billing',mode='private',model_version='tampered')
        self.assertEqual(route(self.model,row),route(self.model,polluted))
        np.testing.assert_array_equal(self.model.features([row]).toarray(),self.model.features([polluted]).toarray())

    def test_policy(self):
        for text,team in [('I paid already, please install the purifier','Installations'),('I paid for this purifier but it is broken and not working','Repairs')]:
            self.assertEqual(route(self.model,{'request_text':text})['team'],team)
        self.assertTrue(route(self.model,{'request_text':'Please call me back'})['review_required'])
        self.assertTrue(route(self.model,{'request_text':'install it and refund my invoice'})['review_required'])

    def test_contract_and_reasons(self):
        row={'request_text':'install purifier technician setup'}
        out=route(self.model,row)
        self.assertEqual(set(out),{'team','display_team','model_version','mode','review_required','model_score','alternatives','reasons'})
        self.assertIn(out['team'],TEAMS)
        self.assertTrue(all(a['team'] in TEAMS for a in out['alternatives']))
        self.assertTrue(any('install' in r for r in out['reasons']))
        self.assertEqual(out['mode'],'demo')
        self.assertTrue(all(out['team'] in r for r in out['reasons'] if r.startswith('Input evidence:')))
        conflict=route(self.model,{'request_text':'install it and refund my invoice'})
        self.assertTrue(any('Human review required' in r for r in conflict['reasons']))
        self.assertFalse(any('customer' in r.casefold() for r in out['reasons']))

    def test_variants_and_persistence(self):
        rows=[{'request_text':t,'channel':'phone'} for t in TEAMS]*2
        for char in [False,True]:
            for metadata in [False,True]:
                m=fit_model(rows,list(TEAMS)*2,{'use_char':char,'use_metadata':metadata})
                self.assertIn(route(m,rows[0])['team'],TEAMS)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pkl';save_model(self.model,p)
            self.assertEqual(route(self.model,rows[0]),route(load_model(p),rows[0]))
            with self.assertRaises(FileNotFoundError):load_model(Path(d)/'missing')

    def test_metadata_explanation_suppresses_shared_boilerplate(self):
        rows=[{'request_text':'shared boilerplate','channel':channel} for channel in ('chat','email') for _ in range(12)]
        model=fit_model(rows,['Repairs']*12+['Billing']*12,{'use_metadata':True,'use_char':False})
        for channel,team in [('chat','Repairs'),('email','Billing')]:
            result=route(model,{'request_text':'shared boilerplate','channel':channel})
            self.assertEqual(result['team'],team)
            evidence=[r for r in result['reasons'] if r.startswith('Input evidence:')]
            self.assertTrue(any(f'channel "{channel}"' in r for r in evidence))
            self.assertFalse(any('shared' in r or 'boilerplate' in r for r in evidence))
            self.assertTrue(all(team in r for r in evidence))

    def test_character_explanation_for_unseen_word(self):
        rows=[{'request_text':text} for text in ('repair','invoice') for _ in range(12)]
        model=fit_model(rows,['Repairs']*12+['Billing']*12,{'use_char':True})
        result=route(model,{'request_text':'repairing'})
        self.assertEqual(result['team'],'Repairs')
        evidence=[r for r in result['reasons'] if r.startswith('Input evidence:')]
        self.assertTrue(any('character fragments' in r for r in evidence))
        self.assertFalse(any('wording' in r for r in evidence))
        self.assertTrue(all('Repairs' in r for r in evidence))
