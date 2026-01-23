python retargeting/retarget.py \
  --model "outputs/character1_to_Karim/2026-01-20/21-16-44/skeletal_gan_epoch1800.pt" \
  --source-skeleton retargeting/data/skeletons/character1.npz \
  --target-skeleton retargeting/data/skeletons/Karim.npz \
  --motion retargeting/data/bandai/processed/character1/dataset-1_walk_active_001.npz \
  --output-dir outputs/retargeting/character1_to_Karim_tiny1