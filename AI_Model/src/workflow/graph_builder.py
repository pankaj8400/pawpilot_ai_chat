from langgraph.graph import START, END, StateGraph
import logging

logger = logging.getLogger(__name__)

from AI_Model.src.workflow.nodes import (
    input_processing_node,
    decision_router_node,
    rag_retrieval_node,
    engineer_prompt_node,
    run_model_inference_node,
    validate_response_node,
    log_interaction_node,
    check_fine_tuning_trigger_node,
)

from AI_Model.src.workflow.state_definition import WorkFlowState


def build_complete_workflow():
    """Build complete LangGraph with all 8 nodes"""
    logger.info("Building workflow...")
    
    # Create state graph
    workflow = StateGraph(WorkFlowState)
    
    # Add all nodes - these are functions that receive Dict and return Dict
    workflow.add_node("input_processing", input_processing_node)
    workflow.add_node("decision_router", decision_router_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("prompt_engineering", engineer_prompt_node)
    workflow.add_node("model_inference", run_model_inference_node)
    workflow.add_node("response_validation", validate_response_node) 
    workflow.add_node("logging", log_interaction_node)               
    workflow.add_node("fine_tuning_check", check_fine_tuning_trigger_node)  
    
    # Define edges
    workflow.add_edge(START, "input_processing")
    workflow.add_edge("input_processing", "decision_router")
    
    # Conditional edge based on use_rag
    def should_use_rag(state):
        """Determine if we should use RAG"""
        return state.get("use_rag", False)
    
    workflow.add_conditional_edges(
        "decision_router",
        should_use_rag,
        {
            True: "rag_retrieval",
            False: "prompt_engineering",
        },
    )
    
    # Rest of the workflow
    workflow.add_edge("rag_retrieval", "prompt_engineering")
    workflow.add_edge("prompt_engineering", "model_inference")
    workflow.add_edge("model_inference", "response_validation")  
    workflow.add_edge("response_validation", "logging")           
    workflow.add_edge("logging", "fine_tuning_check")            
    workflow.add_edge("fine_tuning_check", END)
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    logger.info("✓ Workflow compiled successfully")
    
    return compiled_workflow
