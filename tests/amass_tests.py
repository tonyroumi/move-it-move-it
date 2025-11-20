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
    adapter = AMASSTAdapter(
        data_dir="./data/raw/amass",
        gender="neutral",
        device="cpu"
    )
    
    # Show setup instructions
    print("Setup Instructions:")
    adapter.download()
    print()
    
    # Extract skeleton from body model
    try:
        skeleton = adapter.extract_skeleton()
        print(f"✓ Skeleton extracted from body model")
        print(f"  Joints: {len(skeleton.joint_names)}")
        print(f"  Edges: {len(skeleton.topology)}")
        print(f"  End effectors: {len(skeleton.end_effectors)}")
        print(f"  Height: {skeleton.height:.3f} m")
        print()
        
        # Print first few topology edges
        print("Topology (first 10 edges):")
        for i, (parent, child) in enumerate(skeleton.topology[:10]):
            parent_name = skeleton.joint_names[parent]
            child_name = skeleton.joint_names[child]
            offset = skeleton.offsets[child]
            length = np.linalg.norm(offset)
            print(f"  {parent_name:12s} -> {child_name:12s}: length={length:.4f}m")
        
    except Exception as e:
        print(f"✗ Could not extract skeleton: {e}")
    
    # Test motion extraction
    files = adapter.list_available_files()
    if files:
        print(f"\n✓ Found {len(files)} motion files")
        test_file = files[0]
        print(f"Testing: {Path(test_file).name}")
        
        motion = adapter.extract_motion(test_file)
        print(f"  Frames: {motion.rotations.shape[0]}")
        print(f"  Joints: {motion.rotations.shape[1]}")
        print(f"  FPS: {motion.fps}")
        print(f"  Valid: {adapter.validate_motion(motion)}")
    else:
        print("\n✗ No motion files found")