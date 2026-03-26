from utils.load_data import load_dataset

test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled = load_dataset(
    is_shuffle=False
)

print("len(test_dataloader_unscaled) =", len(test_dataloader_unscaled))