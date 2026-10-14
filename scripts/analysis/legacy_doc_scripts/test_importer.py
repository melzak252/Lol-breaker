import unittest
from unittest.mock import patch
from src.utils import convert_years_to_url
from src.data_extractor import GetData
class Years(unittest.TestCase):
 def test_previous_year_without_current_mapping(self):
  with patch('src.utils.CSV_FILES',{'2025':'old.csv'}):self.assertEqual(convert_years_to_url([2025],2026),['old.csv'])
 def test_missing_year_is_explicit(self):
  with patch('src.utils.CSV_FILES',{'2025':'old.csv'}):
   with self.assertRaisesRegex(ValueError,'2026'):convert_years_to_url([2026],2026)
 def test_default_and_string_year(self):
  with patch('src.utils.CSV_FILES',{'2026':'new.csv'}):
   self.assertEqual(convert_years_to_url(None,2026),['new.csv']);self.assertEqual(convert_years_to_url('2026',2026),['new.csv'])
 def test_years_preserve_order(self):
  with patch('src.utils.CSV_FILES',{'2025':'a','2026':'b'}):self.assertEqual(convert_years_to_url([2026,2025,2026],2026),['b','a'])
 def test_full_import_for_2026_with_local_fixture(self):
  import tempfile
  from pathlib import Path
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'2026.csv'
   path.write_text('gameid,participantid,playername,teamname,year\nfixture-2026,1,synthetic-player,synthetic-team,2026\n')
   with patch('src.utils.CSV_FILES',{'2026':str(path)}), patch('src.data_extractor.CURRENT_YEAR',2026):
    data=GetData(limits=1)
    self.assertEqual(data.df_matches.shape,(1,5))
    self.assertEqual(data.player_names,['synthetic-player'])
    self.assertEqual(int(data.df_matches.iloc[0]['year']),2026)
 def test_rejects_non_oe_response(self):
  import pandas as pd
  with patch('src.data_extractor.convert_years_to_url',return_value=['unused']), patch('src.data_extractor.merge_csv_files',return_value=pd.DataFrame(columns=['html'])):
   with self.assertRaisesRegex(ValueError,'Oracle'):GetData(year=[2026])
 def test_rejects_wrong_year_in_file(self):
  import pandas as pd
  frame=pd.DataFrame({'gameid':['x'],'participantid':[1],'playername':['p'],'teamname':['t'],'year':[2025]})
  with patch('src.data_extractor.convert_years_to_url',return_value=['unused']),patch('src.data_extractor.merge_csv_files',return_value=frame):
   with self.assertRaisesRegex(ValueError,'year'):GetData(year=[2026])
if __name__=='__main__':unittest.main()
