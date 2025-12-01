from src.dataset import MotionDataset, MotionDatasetBuilder
from src.data_processing.adapters.amass import AMASSTAdapter
from src.data_processing.base import DataSourceAdapter
from src.utils.io import load_yaml
from torch.utils.data import Dataset, DataLoader

# builder = MotionDatasetBuilder(
#     adapter=AMASSTAdapter("HUMAN4D", "neutral"),
#     window_size=64,
# )

# dataset = MotionDataset(
#     builder=builder,
#     dataset="human4d",
#     characters=["Aude", "Karim"]
# )

def test_training_loop(loader, dataset):
    print("===== START TEST LOOP =====")

    # Recover char ID → char name mapping (from dataset)
    # id_to_char = dataset.id_to_char

    for batch_idx, batch in enumerate(loader):
        print(f"\n--- Batch {batch_idx} ---")
        print(f"motion.shape  = {batch['motion'].shape}")     # (B, W, F)
        print(f"char_id       = {batch['char_id']}")          # tensor of ints
        print(f"height        = {batch['height']}")           # tensor [B]
        print(f"num_offsets   = {len(batch['offsets'])}")     # B

        # Validate mapping for each sample in the batch
        for i in range(batch["motion"].shape[0]):
            cid = batch["char_id"][i].item()
            cname = id_to_char[cid]

            # metadata stored originally inside dataset.char_meta
            meta = dataset.char_meta[cname]

            print(f"  Sample {i}:")
            print(f"    Character: {cname}")
            print(f"    Height(batch): {batch['height'][i].item():.4f}")
            print(f"    Height(meta):  {meta['height']:.4f}")

            # Validate offsets match (same pointer or same values)
            assert (batch["offsets"][i] == meta["offsets"]).all(), \
                "Offsets do not match metadata!"

            # Validate height matches
            assert abs(batch["height"][i].item() - meta["height"]) < 1e-6, \
                "Height mismatch!"

        # Only test a few batches  
        if batch_idx == 2:
            break

    print("\n===== TEST LOOP DONE =====")

if __name__ == "__main__":

    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")
    
    config = load_yaml('configs/default/motion_dataset.yaml')['GroupA']
    
    
    builder = MotionDatasetBuilder(adapter=AMASSTAdapter(), data_config=config)

    dataset = MotionDataset(builder=builder,
                            characters=["Karim"])
    
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
    )

    test_training_loop(loader, dataset)
    

    
    


    # DataSourceAdapter("amass")