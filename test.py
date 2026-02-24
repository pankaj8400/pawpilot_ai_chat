# --- Usage ---
from AI_Model.src.workflow.nodes import run_model_inference_node
from AI_Model.src.workflow.state_definition import WorkFlowState                
import traceback
if __name__ == "__main__":
    try:
        run_model_inference_node(WorkFlowState())
    except Exception as e:
        print(traceback.format_exc())