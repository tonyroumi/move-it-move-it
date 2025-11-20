from src.dataset import MotionDataset, MotionDatasetBuilder
from src.data_processing.adapters.amass import AMASSTAdapter
from src.data_processing.base import DataSourceAdapter


# builder = MotionDatasetBuilder(
#     adapter=AMASSTAdapter("HUMAN4D", "neutral"),
#     window_size=64,
# )

# dataset = MotionDataset(
#     builder=builder,
#     dataset="human4d",
#     characters=["Aude", "Karim"]
# )

if __name__ == "__main__":

    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")
    
    builder = MotionDatasetBuilder(adapter=AMASSTAdapter(), window_size=64)

    dataset = MotionDataset(builder=builder,
                            characters=["Aude", "Karim"])

    # DataSourceAdapter("amass")