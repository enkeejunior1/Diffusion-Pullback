# Diffusion-Pullback
Official Implementation of understanding the latent space of diffusion models through the lens of riemannian geometry (NeurIPS 2023)

## Environment
```
conda create -n pullback python=3.10
pip install -r requirements.txt
```

## Experiemtns

- unconditional diffusion model

```
bash scripts/main_celeba_hf_local_encoder_pullback.sh
```

- stable diffusion (w/o text condition)

```
bash scripts/main_various_local_encoder_pullback.sh
```

- unconditional diffusion (w/ text condition)

```
bash scripts/main_celeba_hf_local_encoder_pullback.sh
```
