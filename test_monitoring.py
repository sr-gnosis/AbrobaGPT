"""Testes leves e offline do subsistema de monitoramento."""
import csv, tempfile, unittest
from pathlib import Path
from monitoring import RunLogger, load_metrics, braille_plot

class MonitoringTests(unittest.TestCase):
 def test_incremental_and_non_overwrite(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); a=RunLogger(root); a.metric(iteration=1,total_iterations=2,train_loss=1.2,validation_loss=1.3,learning_rate=.001,perplexity=3.7,best_validation_loss=1.3,best_iteration=1,early_stopping=False)
   b=RunLogger(root); b.metric(iteration=2,total_iterations=2,train_loss=1.,validation_loss=1.1,learning_rate=.0001,perplexity=3.,best_validation_loss=1.1,best_iteration=2,early_stopping=False)
   self.assertNotEqual(a.directory,b.directory); self.assertEqual(len(list(root.iterdir())),2)
   with (a.directory/'metrics.csv').open(encoding='utf-8', newline='') as f: self.assertEqual(len(list(csv.DictReader(f))),1)
   self.assertEqual(len((a.directory/'metrics.jsonl').read_text(encoding='utf-8').splitlines()),1)
 def test_render_edge_cases(self):
  cases=[[], [('const',[(1,2),(2,2)],'⣿')], [('train',[(1,3),(2,2)],'⠿'),('val',[(1,4),(2,1)],'⣿')], [('bad',[(1,float('nan'))],'⣿')]]
  for case in cases:
   for width,height in [(80,24),(120,40),(30,10)]:
    lines=braille_plot(case,width,height,'Teste'); self.assertTrue(lines); self.assertTrue(all(isinstance(line,str) for line in lines))
 def test_latest_history(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); logger=RunLogger(root); logger.metric(iteration=3,total_iterations=5,train_loss=1,validation_loss=1,learning_rate=.1,perplexity=2,best_validation_loss=1,best_iteration=3,early_stopping=False)
   d,rows=load_metrics(root); self.assertEqual(d,logger.directory); self.assertEqual(int(rows[0]['iteration']),3)
if __name__=='__main__': unittest.main()
