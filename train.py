import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, NBeats
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import CrossEntropy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

# Load the dataset
df = pd.read_csv("data/processed_solexs_tft_ready.csv", parse_dates=["DATETIME"])
df["time_idx"] = range(len(df))
df["group_id"] = "solarflare"

# Define parameters
encoder_length = 288   # Past 24h (5-min resolution)
decoder_length = 288   # Predict next 24h

# Define TFT-ready dataset
tft_dataset = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="FLARE_TARGET_24h",
    group_ids=["group_id"],
    max_encoder_length=encoder_length,
    max_prediction_length=decoder_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["FLUX_SMOOTH"],
    static_categoricals=["group_id"],
    target_normalizer=NaNLabelEncoder().fit(df["FLARE_TARGET_24h"]),
    allow_missing_timesteps=True,
)

# Train/val split
split_idx = int(0.8 * len(df))
df_train = df.iloc[:split_idx]
df_val = df.iloc[split_idx:]

# Create two separate datasets
train_dataset = TimeSeriesDataSet(
    df_train,
    time_idx="time_idx",
    target="FLARE_TARGET_24h",
    group_ids=["group_id"],
    max_encoder_length=encoder_length,
    max_prediction_length=decoder_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["FLUX_SMOOTH"],
    static_categoricals=["group_id"],
    target_normalizer=NaNLabelEncoder().fit(df["FLARE_TARGET_24h"]),
    allow_missing_timesteps=True,
)

val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, df_val, stop_randomization=True)


# Dataloaders
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=4)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=4)

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4)
checkpoint = ModelCheckpoint(dirpath="outputs/checkpoints", save_top_k=1, monitor="val_loss")
logger = CSVLogger("outputs", name="tft_solarflare")

# Define model
tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=1e-3,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    output_size=1,
    loss=CrossEntropy(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# Train
trainer = Trainer(
    max_epochs=30,
    accelerator="auto",
    callbacks=[early_stop, checkpoint],
    logger=logger,
    gradient_clip_val=0.1,
)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    # Now run training
    trainer.fit(tft, train_dataloader, val_dataloader)

# Save predictions
best_model_path = checkpoint.best_model_path
best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
predictions = best_model.predict(val_dataloader, return_y=True, trainer=trainer)

pred_df = pd.DataFrame({
    "y_pred": predictions.output.flatten(),
    "y_true": predictions.y.flatten()
})
pred_df.to_csv("outputs/tft_predictions.csv", index=False)
print(f"Saved predictions and best model to: {best_model_path}")


