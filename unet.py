from datasets import Dataset
from diffusers import StableDiffusionPipeline
import numpy as np
import torch
from tqdm.auto import tqdm

import os


def load_unet(pipeline: StableDiffusionPipeline, generation: int) -> None:
    """
    Load UNet state from a specified generation into a Stable Diffusion pipeline.

    Args:
        pipeline (`StableDiffusionPipeline`):
            Stable Diffusion pipeline.
        generation (`int`):
            Generation number.
    """

    # Load a saved state dictionary from the generation
    pipeline.unet.load_state_dict(torch.load(f"data/gen_{generation}/unet.pth"))


def save_unet(pipeline: StableDiffusionPipeline, generation: int) -> None:
    """
    Save UNet state within a Stable Diffusion pipeline from a specified generation.

    Args:
        pipeline (`StableDiffusionPipeline`):
            Stable Diffusion pipeline.
        generation (`int`):
            Generation number.
    """

    # Create directory to store state
    os.makedirs(f"data/gen_{generation}", exist_ok=True)

    # Save the state dictionary from the generation
    torch.save(pipeline.unet.state_dict(), f"data/gen_{generation}/unet.pth")


def train_unet(
        pipeline: StableDiffusionPipeline,
        dataset: Dataset,
        epochs: int = 50,
        batch_size: int = 25,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
) -> np.ndarray:
    """
    Training loop for a UNet within a Stable Diffusion pipeline.

    Args:
        pipeline (`StableDiffusionPipeline`):
            Stable Diffusion pipeline.
        dataset (`Dataset`):
            Dataset to train on.
        epochs (`int`, defaults fo `50`):
            Number of epochs.
        batch_size (`int`, defaults to `25`):
            Batch size.
        lr (`float`, defaults to `1e-3`):
            Learning rate.
        weight_decay (`float`, defaults to `1e-2`):
            Weight decay.
    
    Returns:
        `np.ndarray`: Training loss.
    """

    # Data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # SGD optimizer
    optimizer = torch.optim.SGD(pipeline.unet.parameters(), lr=lr, weight_decay=weight_decay)

    # Set discrete timesteps
    pipeline.scheduler.set_timesteps(pipeline.scheduler.config.num_train_timesteps)

    # Store training loss
    train_loss = []

    # Training loop
    for epoch in range(epochs):

        # Set UNet in training mode
        pipeline.unet.train()

        # Iterate through batches
        for batch in tqdm(dataloader, desc=f"Epoch: {epoch}"):

            # Reset gradients
            optimizer.zero_grad()

            # Batch features
            pixel_values = batch["pixel_values"].to("cuda", torch.float16)
            input_ids = batch["input_ids"].to("cuda")

            # Convert images to latent space
            with torch.no_grad():
                latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor

            # Sample noise to add to latents
            noise = torch.randn_like(latents)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                pipeline.scheduler.config.num_train_timesteps,
                (latents.shape[0], ),
                device="cuda",
            ).long()

            # Add noise to latents
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

            # Get text embedding
            with torch.no_grad():
                encoder_hidden_states = pipeline.text_encoder(input_ids).last_hidden_state

            # Predict noise
            pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Calculate noise residual
            loss = torch.nn.functional.mse_loss(pred, noise)

            # Save and display loss
            train_loss.append(loss.item())
            print(f"Loss: {loss.item()}")

            # Compute gradients
            loss.backward()

            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(pipeline.unet.parameters(), 1)

            # Perform optimization
            optimizer.step()

    # Return training loss
    return np.array(train_loss)


def save_train_loss(train_loss: np.ndarray, generation: int) -> None:
    """
    Save a NumPy array containing training loss for a specified generation.

    Args:
        train_loss (`np.ndarray`):
            Training loss.
        generation (`int`):
            Generation number.
    """

    # Create directory to store training loss
    os.makedirs(f"data/gen_{generation}", exist_ok=True)

    # Save training loss
    np.save(f"data/gen_{generation}/train_loss.npy", train_loss)
