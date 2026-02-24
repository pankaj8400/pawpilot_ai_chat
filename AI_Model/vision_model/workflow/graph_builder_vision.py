from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing import cast
import logging
import time
import sys
from AI_Model.vision_model.workflow.nodes_vision import (
    input_processing_node,
    decision_router_node,   
    model_call_node,
    second_model_node,
    retrieval_node,
)

from AI_Model.src.utils.exceptions import CustomException

logger = logging.getLogger(__name__)

# Import nodes for subgraph B (Workflow Engineering)
from AI_Model.src.workflow.nodes import (
    engineer_prompt_node,
    run_model_inference_node,
    validate_response_node,
    log_interaction_node,
    check_fine_tuning_trigger_node
)

# Import state definitions
from AI_Model.src.workflow.state_definition import WorkFlowState, create_initial_state as create_workflow_state
from AI_Model.vision_model.workflow.state_definition_vision import VisionWorkFlowState, create_initial_state as create_vision_state


class StateTransformer:
    """Handles state transformations between VisionWorkFlowState and WorkFlowState"""
    
    @staticmethod
    def vision_to_workflow(vision_state: VisionWorkFlowState) -> WorkFlowState:
        """
        Transform VisionWorkFlowState output to WorkFlowState input.
        
        Maps vision model outputs to workflow state fields.
        """
        try:
            logger.info("Transforming VisionWorkFlowState to WorkFlowState")
            
            workflow_state = create_workflow_state(
                query=vision_state.get("query", ""),
                user_id=vision_state.get("user_id", "user_1"),
                session_id=vision_state.get("session_id", ""),
                use_rag=vision_state.get("use_rag", True),
                model_to_use=vision_state.get("model_to_use", "gpt-4-turbo"),
                strategy=vision_state.get("strategy", "prompt_only")
            )
            if vision_state.get("strategy") in ("emotion-detection", "full-body-scan"):
                workflow_state["to_use_model"] = False
                workflow_state['validated_response'] = vision_state.get("final_output", "")
            # Map vision-specific outputs to workflow inputs
            workflow_state["context"] = f"""
Vision Model Analysis:
- Predicted Class: {vision_state.get("predicted_class", "Unknown")}
- Confidence Score: {vision_state.get("confidence_score", 0.0):.2f}
- meta data retrieved: {vision_state.get("retrieved_docs", [])}
- Model 2 Response: {vision_state.get("raw_model2_response", "")}
"""
            workflow_state["predicted_class"] = vision_state.get("predicted_class", "Unknown")
            # Carry over retrieved docs if any
            if vision_state.get("retrieved_docs"):
                workflow_state["retrieved_docs"] = [
                    {"content": str(doc)} for doc in vision_state.get("retrieved_docs", {}).values()
                ]
            
            # Use vision model's inference time
            if vision_state.get("inference_time"):
                workflow_state["inference_time"] = vision_state.get("inference_time", 0.0)
            
            # Preserve timing info
            workflow_state["start_time"] = vision_state.get("start_time", time.time())
            
            # Carry forward any errors
            if vision_state.get("error"):
                workflow_state["error"] = vision_state.get("error", [])
            
            logger.info("State transformation completed successfully")
            return workflow_state
            
        except Exception as e:
            logger.error(f"Error transforming state: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def workflow_to_final_output(workflow_state: WorkFlowState) -> dict:
        """
        Transform final WorkFlowState to output format combining both pipelines.
        """
        try:
            logger.info("Creating final output from workflow state")
            return {
                "user_id": workflow_state.get("user_id"),
                "session_id": workflow_state.get("session_id"),
                "vision_output": workflow_state.get("context", ""),
                "final_response": workflow_state.get("validated_response", ""),
                "confidence_score": workflow_state.get("confidence_score", 0.0),
                "citations": workflow_state.get("citations", []),
                "metrics": {
                    "total_time": workflow_state.get("total_time", 0.0),
                    "inference_time": workflow_state.get("inference_time", 0.0),
                    "cost": workflow_state.get("cost", 0.0),
                },
                "error": workflow_state.get("error", [])
            }
            
        except Exception as e:
            logger.error(f"Error creating final output: {str(e)}")
            raise CustomException(e, sys)


class MultiGraphWorkflow:
    """Manages composition of Vision and Workflow StateGraphs with automatic state transformation"""
    
    def __init__(self):
        self.subgraph_a = None
        self.subgraph_b = None
        self.parent_graph = None
        self.state_transformer = StateTransformer()
        
    def build_subgraph_a(self) -> CompiledStateGraph:
        """
        Build first subgraph (Vision Processing) - Nodes 01-05
        Uses: VisionWorkFlowState
        Nodes: input_processing, decision_router, model_call, second_model, retrieval
        """
        try:
            logger.info("Building Subgraph A (Vision Processing)")
            
            graph_a = StateGraph(VisionWorkFlowState)
            
            # Add nodes (01-05)
            graph_a.add_node("input_processing", input_processing_node)      # 01
            graph_a.add_node("decision_router", decision_router_node)        # 02
            graph_a.add_node("model_call", model_call_node)                  # 03
            graph_a.add_node("second_model", second_model_node)              # 04
            graph_a.add_node("retrieval", retrieval_node)                    # 05
            
            # Define edges for subgraph A
            graph_a.add_edge(START, "input_processing")
            graph_a.add_edge("input_processing", "decision_router")
            graph_a.add_edge("decision_router", "model_call")
            graph_a.add_edge("model_call", "retrieval")
            graph_a.add_edge("retrieval", "second_model")
            graph_a.add_edge("second_model", END)
            
            self.subgraph_a = graph_a.compile()
            logger.info("Subgraph A built successfully with nodes 01-05")
            return self.subgraph_a
            
        except Exception as e:
            logger.error(f"Error building subgraph A: {str(e)}")
            raise CustomException(e, sys)
    
    def build_subgraph_b(self) -> CompiledStateGraph:
        """
        Build second subgraph (Workflow Engineering) - Nodes 06-10
        Uses: WorkFlowState
        Nodes: engineer_prompt, run_inference, validate_response, log_interaction, check_fine_tuning
        """
        try:
            logger.info("Building Subgraph B (Workflow Engineering)")
            
            graph_b = StateGraph(WorkFlowState)
            
            # Add nodes (06-10)
            graph_b.add_node("engineer_prompt", engineer_prompt_node)        # 06
            graph_b.add_node("run_inference", run_model_inference_node)      # 07
            graph_b.add_node("validate_response", validate_response_node)    # 08
            graph_b.add_node("log_interaction", log_interaction_node)        # 09
            graph_b.add_node("check_fine_tuning", check_fine_tuning_trigger_node)  # 10
            
            # Define edges for subgraph B
            graph_b.add_edge(START, "engineer_prompt")
            graph_b.add_edge("engineer_prompt", "run_inference")
            graph_b.add_edge("run_inference", "validate_response")
            graph_b.add_edge("validate_response", "log_interaction")            
            graph_b.add_edge("log_interaction", "check_fine_tuning")
            graph_b.add_edge("check_fine_tuning", END)
            
            self.subgraph_b = graph_b.compile()
            logger.info("Subgraph B built successfully with nodes 06-10")
            return self.subgraph_b
            
        except Exception as e:
            logger.error(f"Error building subgraph B: {str(e)}")
            raise CustomException(e, sys)
    
    def build_parent_graph(self) -> CompiledStateGraph:
        """
        Build parent graph orchestrating both subgraphs.
        Handles automatic state transformation between VisionWorkFlowState and WorkFlowState.
        
        Flow: VisionSubgraph (A) -> Transform -> WorkflowSubgraph (B) -> Final Output
        """
        try:
            logger.info("Building parent workflow graph with state transformation")
            
            if not self.subgraph_a or not self.subgraph_b:
                self.build_subgraph_a()
                self.build_subgraph_b()
            
            # Ensure subgraphs are built
            assert self.subgraph_a is not None, "Subgraph A failed to build"
            assert self.subgraph_b is not None, "Subgraph B failed to build"
            
            # Parent graph uses WorkFlowState (the more comprehensive state)
            parent_graph = StateGraph(WorkFlowState)
            
            # Add compiled subgraphs as nodes
            parent_graph.add_node("vision_subgraph", self.subgraph_a)
            parent_graph.add_node("workflow_subgraph", self.subgraph_b)
            
            # Add state transformation node
            parent_graph.add_node(
                "transform_state",
                self._transform_state_node
            )
            
            # Define flow: Start -> Vision -> Transform -> Workflow -> End
            parent_graph.add_edge(START, "vision_subgraph")
            parent_graph.add_edge("vision_subgraph", "transform_state")
            parent_graph.add_edge("transform_state", "workflow_subgraph")
            parent_graph.add_edge("workflow_subgraph", END)
            
            self.parent_graph = parent_graph.compile()
            logger.info("Parent workflow graph built successfully")
            return self.parent_graph
            
        except Exception as e:
            logger.error(f"Error building parent graph: {str(e)}")
            raise CustomException(e, sys)
    
    def _transform_state_node(self, state: WorkFlowState) -> WorkFlowState:
        """
        Internal node that transforms state between subgraphs.
        This node sits between vision_subgraph output and workflow_subgraph input.
        """
        try:
            logger.info("Executing state transformation node")
            
            # The state coming in is already WorkFlowState from vision output
            # But we may need to enrich it with transformed vision data
            if "context" not in state or not state["context"]:
                # Vision results should already be in context, but ensure it's there
                state["context"] = state.get("context", "")
                state["predicted_class"] = state.get("predicted_class", "Unknown")
            return state
            
        except Exception as e:
            logger.error(f"Error in state transformation node: {str(e)}")
            raise CustomException(e,sys)
    
    @staticmethod
    def _route_decision_a(state: VisionWorkFlowState) -> str:
        """Router for subgraph A decision point"""
        try:
            # Route based on strategy or other conditions
            strategy = state.get("strategy", "default")
            
            if strategy == "retrieval_only":
                return "retrieval"
            elif strategy == "model_only":
                return "model_call"
            else:
                return "model_call"  # Default to model_call
                
        except Exception as e:
            logger.error(f"Error in decision routing: {str(e)}")
            return "model_call"
    
    @staticmethod
    def _route_validation_b(state: WorkFlowState) -> str:
        """Router for subgraph B validation point"""
        try:
            # Route based on validation result
            if state.get("validated_response") and state.get("confidence_score", 0) > 0.5:
                return "valid"
            return "invalid"
            
        except Exception as e:
            logger.error(f"Error in validation routing: {str(e)}")
            return "invalid"
    
    def get_workflow(self) -> CompiledStateGraph:
        """Get the compiled parent workflow"""
        if self.parent_graph is None:
            self.build_parent_graph()
        assert self.parent_graph is not None, "Parent graph failed to build"
        return self.parent_graph
    

    def invoke(self, input_data: dict):
        """
        Execute the workflow with given input.
        
        Args:
            input_data: Dictionary with 'image', 'query', and other vision inputs
            
        Returns:
            Final combined output from both pipelines
        """
        try:
            logger.info("Invoking multi-stage workflow")
            
            # Create initial vision state
            vision_state = create_vision_state(
                query=input_data.get("query", ""),
                user_id=input_data.get("user_id", "user_1"),
                session_id=input_data.get("session_id", ""),
                model_to_use=input_data.get("model_to_use", "diseases_classifier"),
                strategy=input_data.get("strategy", "default")
            )
            
            # Add image if provided
            if "image" in input_data:
                vision_state["image"] = input_data["image"]
            
            # Ensure subgraphs are built
            if self.subgraph_a is None:
                self.subgraph_a = self.build_subgraph_a()
            if self.subgraph_b is None:
                self.subgraph_b = self.build_subgraph_b()
            
            # Execute vision subgraph FIRST with vision_state
            vision_workflow = self.subgraph_a
            assert vision_workflow is not None, "Subgraph A failed to build"
            vision_result = vision_workflow.invoke(vision_state)
            
            logger.info(f"Vision subgraph completed. Predicted class: {vision_result.get('predicted_class')}")
            
            # Transform vision result to workflow state
            workflow_state = self.state_transformer.vision_to_workflow(cast(VisionWorkFlowState, vision_result))
            
            # Execute workflow subgraph with transformed state
            assert self.subgraph_b is not None, "Subgraph B failed to build"
            workflow_result = self.subgraph_b.invoke(workflow_state)
            
            logger.info("Workflow completed successfully")
            
            # Transform to final output
            final_output = self.state_transformer.workflow_to_final_output(cast(WorkFlowState, workflow_result))
            return final_output
            
        except Exception as e:
            logger.error(f"Error during workflow execution: {str(e)}")
            raise CustomException(e, sys)


    def stream(self, input_data: dict):
        """
        Stream workflow execution with intermediate results. 
        
        Args:
            input_data: Dictionary with input parameters
            
        Yields:
            Intermediate state updates and final result
        """
        try:
            logger.info("Streaming multi-stage workflow execution")
            
            vision_state = create_vision_state(
                query=input_data.get("query", ""),
                user_id=input_data.get("user_id", "user_1"),
                session_id=input_data.get("session_id", ""),
                model_to_use=input_data.get("model_to_use", "diseases_classifier"),
                strategy=input_data.get("strategy", "default")
            )
            
            if "image" in input_data:
                vision_state["image"] = input_data["image"]
            
            # Ensure subgraph_a is built
            if self.subgraph_a is None:
                self.subgraph_a = self.build_subgraph_a()
            # Stream vision subgraph first
            for event in self.subgraph_a.stream(vision_state):
                logger.debug(f"Vision stream event: {event}")
                yield {"stage": "vision", "event": event}
            
            # Get final vision result
            vision_result = self.subgraph_a.invoke(vision_state)
            
            # Transform and stream workflow subgraph
            workflow_state = self.state_transformer.vision_to_workflow(cast(VisionWorkFlowState, vision_result))
            
            if self.subgraph_b is None:
                self.build_subgraph_b()
            assert self.subgraph_b is not None, "Subgraph B failed to build"
            for event in self.subgraph_b.stream(workflow_state):
                logger.debug(f"Workflow stream event: {event}")
                yield {"stage": "workflow", "event": event}
                
        except Exception as e:
            logger.error(f"Error during workflow streaming: {str(e)}")
            raise CustomException(e, sys)


# Usage example
if __name__ == "__main__":
    try:
        # Initialize workflow
        workflow_manager = MultiGraphWorkflow()
        
        # Build all graphs
        workflow_manager.build_subgraph_a()
        workflow_manager.build_subgraph_b()
        workflow_manager.build_parent_graph()
        
        logger.info("All subgraphs and parent graph built successfully")
        
        # Prepare input data (with image)
        import numpy as np
        initial_input = {
            "query": "Which toy is this ?",
            "user_id": "user_123",
            "session_id": "session_abc",
            "image": 'AI_Model/vision_model/data/Dental Chew Toy.png',  # Example image
            "model_to_use": "toy_classifier",
            "strategy": "default"
        }
        
        # Execute workflow
        result = workflow_manager.invoke(initial_input)
        print("\n✅ Workflow Result:")
        print(f"User ID: {result['user_id']}")
        print(f"Session ID: {result['session_id']}")
        print(f"Vision Output: {result['vision_output']}")
        print(f"Final Response: {result['final_response']}")
        print(f"Confidence: {result['confidence_score']}")
        print(f"Total Time: {result['metrics']['total_time']:.2f}s")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
