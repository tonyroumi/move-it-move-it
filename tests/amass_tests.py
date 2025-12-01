from src.data_processing.adapters.amass import AMASSTAdapter
from pathlib import Path

# Example usage
if __name__ == "__main__":

    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")

    # Initialize adapter with body model (simplified - no shape parameters)
    adapter = AMASSTAdapter()
    
    # Show setup instructions
    # print("Setup Instructions:")
    # adapter.download()
    # print()
    
    # Extract skeleton and motion
    # try:
    #     # motion = adapter.extract_motion("Aude")
    #     skeleton = adapter.extract_skeleton("Carine")
    #     print(f"✓ Skeleton extracted from body model")
    #     print(f"  Edges: {len(skeleton.topology)}")
    #     print(f"  End effectors: {len(skeleton.ee_ids)}")
    #     print(f"  Height: {skeleton.height:.3f} m")
    #     print()
        
        
    # except Exception as e:
    #     print(f"✗ Could not extract skeleton: {e}")
    
    try:        
        motion = adapter.extract_motion("Carine")
        print(f"  Frames: {motion.rotations.shape[0]}")
        print(f"  Joints: {motion.rotations.shape[1]}")
        print(f"  FPS: {motion.fps}")
        print(f"  Valid: {adapter.validate_motion(motion)}")
    except Exception as e:
        print(f"✗ Could not extract motion: {e}")