from vart import Runner
import xir
import numpy as np
from config import Config

class FPGAEnsemble:
    def __init__(self, xmodel_paths=Config.XMODEL_PATHS):
        self.runners = []
        for path in xmodel_paths:
            g = xir.Graph.deserialize(path)
            sg = g.get_root_subgraph().toposort_child_subgraph()[0]
            self.runners.append(Runner.create_runner(sg, "run"))

    def predict(self, X):
        outputs = []
        for runner in self.runners:
            out = np.empty((1, 9), dtype=np.int8)
            job_id = runner.execute_async(X.astype(np.int8), out)
            runner.wait(job_id)
            outputs.append(out.astype(np.float32))
        final_pred = np.mean(np.vstack(outputs), axis=0)
        return int(final_pred.argmax())
