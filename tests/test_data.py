import csv
import tempfile
import unittest
from pathlib import Path
from routing.data import canonical_team, clean_text, input_signature, load_pack, audit_pack

TEAMS = ['Installations','Repairs','Consumables','Billing','Returns & Replacement','Warranty Claims','Product Advice']

class DataTests(unittest.TestCase):
    def pack(self, path):
        intake = {'created_at_ist':'2025-01-01 10:00','channel':'chat','product_family':'Vacuum','warranty_status':'shield','source':'crm'}
        files = {'teams.csv': [{'team': x, 'renamed_to': '', 'handles':'Synthetic remit'} for x in TEAMS], 'train.csv':[dict(intake, request_id='a',request_text='Order ORD123 broken',team_label='Repairs')], 'test_unlabelled.csv':[dict(intake,request_id='b',request_text='help')], 'resolution_log.csv':[{'request_id':'a','first_team':'Repairs','final_team':'Repairs','transfers':'0','resolved_at':'2025-01-01 11:00'}], 'sample_submission.csv':[{'request_id':'b','team':'Repairs'}]}
        for name, rows in files.items():
            with (path/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        return files
    def test_missing_full_schema_fields(self):
        cases={'train.csv':('channel','product_family','warranty_status','source','created_at_ist'), 'test_unlabelled.csv':('channel','product_family','warranty_status','source','created_at_ist'), 'resolution_log.csv':('first_team','transfers','resolved_at'), 'teams.csv':('renamed_to','handles')}
        for filename, fields in cases.items():
            for field in fields:
                with self.subTest(filename=filename,field=field), tempfile.TemporaryDirectory() as d:
                    p=Path(d);files=self.pack(p);rows=files[filename]
                    for row in rows:del row[field]
                    with (p/filename).open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
                    with self.assertRaisesRegex(ValueError,'missing required columns'):load_pack(p)
    def test_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.pack(p);file=p/'train.csv';lines=file.read_text().splitlines()
            lines[0]+=',channel';lines[1]+=',email';file.write_text('\n'.join(lines)+'\n')
            with self.assertRaisesRegex(ValueError,'duplicate headers'):load_pack(p)
    def test_only_rename_column_may_be_blank(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.pack(p);file=p/'teams.csv';file.write_text(file.read_text().replace('Synthetic remit',''))
            with self.assertRaisesRegex(ValueError,'blank or malformed'):load_pack(p)
    def test_aliases_and_unknown(self):
        self.assertEqual(canonical_team('Installs & Demo'),'Installations')
        self.assertEqual(canonical_team('Filters & Consumables'),'Consumables')
        with self.assertRaisesRegex(ValueError,'Unknown team'): canonical_team('Other')
    def test_signatures_and_no_mutation(self):
        row={'request_text':'  BROKEN order ORD123 req SR12345  ','product_family':'Vacuum','warranty_status':'shield'}
        original=dict(row)
        self.assertEqual(input_signature(row),input_signature(dict(row,request_text='broken order ORD999 req SR99999')))
        self.assertEqual(row,original)
        self.assertEqual(clean_text('hello\n WORLD'),'hello world')
    def test_audit_corrects_legacy_only_without_mutation(self):
        row={'request_id':'a','request_text':'broken','product_family':'Vacuum','warranty_status':'shield','team_label':'Repairs','created_at_ist':'2025-01-01 10:00','source':'legacy_zoho'}
        outcome={'request_id':'a','final_team':'Repairs','resolved_at':'2025-01-01 05:00'}
        pack={'train':[row],'test':[], 'resolution':[outcome]}
        self.assertEqual(audit_pack(pack)['corrected_negative_elapsed'],0)
        self.assertEqual(outcome['resolved_at'],'2025-01-01 05:00')
        row['source']='crm'
        self.assertEqual(audit_pack(pack)['corrected_negative_elapsed'],1)
    def test_join_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); self.pack(p)
            file=p/'resolution_log.csv';file.write_text(file.read_text().replace('a,Repairs','z,Repairs'))
            with self.assertRaisesRegex(ValueError,'resolution.*coverage'):load_pack(p)
    def test_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.pack(p)
            file=p/'train.csv'
            with file.open('a') as f:f.write(file.read_text().splitlines()[1]+'\n')
            with self.assertRaisesRegex(ValueError,'Duplicate request_id'):load_pack(p)
    def test_order_and_original_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.pack(p);pack=load_pack(p)
            self.assertEqual(pack['train'][0]['request_text'],'Order ORD123 broken')
            (p/'sample_submission.csv').write_text('request_id,team\nx,Repairs\n')
            with self.assertRaisesRegex(ValueError,'template.*order'):load_pack(p)

if __name__=='__main__':unittest.main()
