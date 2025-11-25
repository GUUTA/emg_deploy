from vart import Runner
import xir
import numpy as np
from config import Config


class FPGAEnsemble:
    def __init__(self, xmodel_paths=Config.XMODEL_PATHS):
        """
        Loads one or more .xmodel files for ensemble inference.
        """
        self.runners = []
        for path in xmodel_paths:
            # Load graph
            graph = xir.Graph.deserialize(path)

            # Identify DPU subgraph
            root = graph.get_root_subgraph()
            subgraphs = root.toposort_child_subgraph()
            dpu_subgraph = subgraphs[0]

            # Create runner
            runner = Runner.create_runner(dpu_subgraph, "run")
            self.runners.append(runner)

    def predict(self, X):
        """
        X shape must be: (1, 700, 10, 1)
        dtype: int8 for DPU
        Returns class index (0–8)
        """
        outputs = []

        # Prepare input as int8
        X_int8 = X.astype(np.int8)

        # Each model produces 9 logits
        for runner in self.runners:
            out = np.empty((1, 9), dtype=np.int8)

            # Execute on DPU
            job_id = runner.execute_async(X_int8, out)
            runner.wait(job_id)

            # Convert fp32 for ensemble averaging
            outputs.append(out.astype(np.float32))

        # Stack all model outputs → average
        final_pred = np.mean(np.vstack(outputs), axis=0)

        # Return class index
        return int(final_pred.argmax())
