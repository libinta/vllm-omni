#!/usr/bin/env python3  
"""  
Complete offline script using OmniEngineArgs directly for Qwen2.5-3B.  
This demonstrates low-level engine initialization without the Omni orchestrator.  
"""  
  
import os  
from pathlib import Path  
  
# Set environment variables for multiprocessing  
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

os.environ["VLLM_SKIP_WARMUP"] = "true"  
from vllm import SamplingParams  
from vllm.assets.image import ImageAsset  
from vllm.v1.engine.llm_engine import LLMEngine  
from vllm_omni.engine.arg_utils import OmniEngineArgs  
  
  
def create_omni_engine():  
    """Create and configure an LLM engine using OmniEngineArgs."""  
    # Create OmniEngineArgs with explicit configuration for Qwen2.5-3B  
    engine_args = OmniEngineArgs(  
        model="Qwen/Qwen2.5-Omni-3B",  
        stage_id=0,  # Thinker stage for image understanding  
        model_stage="thinker",  
        model_arch="Qwen2_5OmniForConditionalGeneration",  
        engine_output_type="text",  # Direct text output  
        hf_config_name=None,  # Use default config for 3B model  
        gpu_memory_utilization=0.8,  
        enforce_eager=True,  # Required for Qwen2.5-Omni  
        trust_remote_code=True,  
        distributed_executor_backend="mp",  
        enable_prefix_caching=False,  
        tensor_parallel_size=1,  
        max_num_batched_tokens=32768,  
        disable_log_stats=True,  
    )  
      
    # Create engine config - this registers omni models  
    vllm_config = engine_args.create_engine_config()  
      
    # Create and return the engine  
    engine = LLMEngine.from_engine_args(engine_args)  
    return engine  
  
def test_image_to_text(engine):  
    """Test image input with text generation."""  
    print("Testing image-to-text generation...")  
      
    # Load test image  
    image = ImageAsset(name="stop_sign").pil_image  
      
    # Prepare multimodal prompt  
    prompt = {  
        "prompt": "Describe this image in detail.",  
        "multi_modal_data": {"image": image}  
    }  
      
    # Create sampling parameters  
    sampling_params = SamplingParams(  
        temperature=0.4,  
        top_p=0.9,  
        max_tokens=2048,  
        seed=42  
    )  
      
    # Add request to engine with correct positional arguments  
    request_id = "test_request_001"  
    engine.add_request(request_id, prompt, sampling_params)  
      
    # Generate response  
    outputs = []  
    while engine.has_unfinished_requests():  
        step_outputs = engine.step()  
        for output in step_outputs:  
            if output.finished and output.request_id == request_id:  
                outputs.append(output)  
      
    # Extract and print text output  
    if outputs:  
        text_output = outputs[0].outputs[0].text  
        print(f"Generated text: {text_output}")  
        return text_output  
    else:  
        print("No output generated")  
        return None
  
  
def main():  
    """Main function demonstrating OmniEngineArgs usage."""  
    print("Qwen2.5-Omni-3B with OmniEngineArgs")  
    print("=" * 40)  
      
    engine = None  
    try:  
        # Create engine using OmniEngineArgs  
        engine = create_omni_engine()  
        print("Engine created successfully")  
          
        # Run image-to-text test  
        result = test_image_to_text(engine)  
          
        if result:  
            print("\nTest completed successfully!")  
        else:  
            print("\nTest failed to generate output")  
              
    except Exception as e:  
        print(f"Error: {e}")  
        raise  
    finally:  
        # Clean up engine resources  
        if engine is not None:  
            if hasattr(engine, 'shutdown'):  
                engine.shutdown()  
            print("Engine shutdown complete")  
  
  
if __name__ == "__main__":  
    main()