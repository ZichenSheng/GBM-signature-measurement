import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
p=Path(__file__).resolve().parents[1]/"analysis/ivy/10_run_analysis.py"
s=importlib.util.spec_from_file_location("ivy",p);A=importlib.util.module_from_spec(s);s.loader.exec_module(A)
class PrecisionRegression(unittest.TestCase):
 def test_equal_reference_statistics_are_counted(self):
  # 0.7 rounds down in float32 by ~1.2e-8, far larger than the retained 1e-12 tolerance.
  self.assertLess(float(np.float32(.7)),.7-1e-12)
  with patch.object(A,"Q2_NPERM",2),patch.object(A,"grouped_predictions",return_value=np.zeros((2,1),dtype=int)),patch.object(A,"classification_metrics",return_value=(np.array([.7]),np.array([.7]),np.ones((2,2,1),dtype=int))),patch.object(A,"permute_within_tumor",side_effect=lambda l,t,r:l):
   result,_=A.q2_identifiability(np.ones((2,1)),np.array(["PAN","MVP"]),np.array(["t","t"]),["PAN","MVP"],np.random.default_rng(1))
  self.assertEqual(float(result.permutation_p.iloc[0]),1.)
if __name__=="__main__":unittest.main()
