import math

import torch
import torch.nn.functional as F
from torch import nn


def make_beta_schedule(schedule: str, num_timesteps: int) -> torch.Tensor:
    """
    Build the variance (beta) schedule for the forward diffusion process.

    :param schedule: Either "linear" (DDPM) or "cosine" (improved DDPM).
    :param num_timesteps: Number of diffusion steps T.
    :return: Tensor of betas, shape (num_timesteps,).
    """
    if schedule == "linear":
        scale = 1000 / num_timesteps
        return torch.linspace(scale * 1e-4, scale * 0.02, num_timesteps, dtype=torch.float64)

    if schedule == "cosine":
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(max=0.999)

    raise ValueError(f"Unknown beta schedule: {schedule}")


def extract(values: torch.Tensor, timestep: torch.Tensor, broadcast_shape: torch.Size) -> torch.Tensor:
    """Gather per-sample scalars from a 1D buffer and reshape for broadcasting over images."""
    out = values.gather(0, timestep).float()
    return out.reshape(timestep.shape[0], *([1] * (len(broadcast_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """
    DDPM forward/reverse process with epsilon-prediction and classifier-free guidance.

    All schedule tensors are registered as buffers so they move with `.to(device)`.
    """

    def __init__(self, num_timesteps: int = 1000, beta_schedule: str = "linear"):
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = make_beta_schedule(beta_schedule, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Posterior q(x_{t-1} | x_t, x_0) variance.
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        def register(name: str, tensor: torch.Tensor) -> None:
            self.register_buffer(name, tensor.float())

        register("betas", betas)
        register("alphas_cumprod", alphas_cumprod)
        register("alphas_cumprod_prev", alphas_cumprod_prev)
        register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        register("posterior_variance", posterior_variance)
        register("posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20)))
        register("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Diffuse x_start to timestep t: x_t = sqrt(a_bar) x_0 + sqrt(1 - a_bar) noise."""
        return (
            extract(self.sqrt_alphas_cumprod, timestep, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timestep, x_start.shape) * noise
        )

    def training_loss(self, model: nn.Module, x_start: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sample a random timestep per image and return the epsilon-prediction MSE."""
        batch = x_start.shape[0]
        timestep = torch.randint(0, self.num_timesteps, (batch,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noised = self.q_sample(x_start, timestep, noise)
        predicted_noise = model(x_noised, timestep, labels)
        return F.mse_loss(predicted_noise, noise)

    def adaptive_cfg_loss(self, model, controller, x_start, labels,
                          classifier=None, classification_weight=0.01,
                          classification_min_snr=1.0, timestep=None, noise=None):
        '''Noise MSE plus gated CE on predicted x0; classifier accepts [-1, 1] images.

        The classifier must be frozen and in eval mode, but its input gradient
        stays enabled. Masked CE is averaged over the full batch, like noise MSE.
        '''
        for name, value in (('classification_weight', classification_weight),
                            ('classification_min_snr', classification_min_snr)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative.')
        if classification_weight > 0 and classifier is None:
            raise ValueError('A frozen classifier is required for classification loss.')
        batch = x_start.shape[0]
        if timestep is None:
            timestep = torch.randint(0, self.num_timesteps, (batch,), device=x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noised = self.q_sample(x_start, timestep, noise)
        null_labels = torch.full_like(labels, model.num_classes)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                prediction = model(
                    torch.cat([x_noised, x_noised]),
                    torch.cat([timestep, timestep]),
                    torch.cat([labels, null_labels]),
                )
        finally:
            model.train(was_training)

        eps_cond, eps_uncond = prediction.chunk(2, dim=0)
        scale_shape = (batch,) + (1,) * (x_start.ndim - 1)
        scale = controller(timestep).reshape(scale_shape)
        guided = eps_uncond + scale * (eps_cond - eps_uncond)
        reconstruction = F.mse_loss(guided, noise)

        classification = reconstruction.new_zeros(())
        if classification_weight > 0:
            alpha_bar = self.alphas_cumprod[timestep]
            snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-12)
            eligible = snr >= classification_min_snr
            if eligible.any():
                # Reconstruct only eligible samples: high-noise x0 is unreliable.
                predicted_x_start = self.predict_x_start_from_noise(
                    x_noised[eligible], timestep[eligible], guided[eligible],
                )
                logits = classifier(predicted_x_start)
                classification = F.cross_entropy(
                    logits, labels[eligible], reduction='sum',
                ) / batch
        total = reconstruction + classification_weight * classification
        return total, reconstruction.detach(), classification.detach()

    def predict_x_start_from_noise(self, x_t, timestep, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, timestep, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timestep, x_t.shape) * noise
        )

    @staticmethod
    def _guidance_scale(timestep, cfg_scale, adaptive_cfg, ndim):
        if adaptive_cfg is None:
            return cfg_scale
        shape = (timestep.shape[0],) + (1,) * (ndim - 1)
        return adaptive_cfg(timestep).reshape(shape)

    @staticmethod
    def _is_conditional_only(scale):
        """True when guidance is a no-op, so the unconditional pass can be skipped."""
        if torch.is_tensor(scale):
            return bool(torch.all(scale == 1.0))
        return scale <= 1.0

    @staticmethod
    def rescale_cfg(guided, conditional, cfg_rescale):
        '''Match guided per-sample std to conditional std, then blend.'''
        if not math.isfinite(cfg_rescale) or not 0.0 <= cfg_rescale <= 1.0:
            raise ValueError('cfg_rescale must be finite and lie in [0, 1].')
        if cfg_rescale == 0.0:
            return guided

        dimensions = tuple(range(1, guided.ndim))
        guided_std = guided.std(dim=dimensions, keepdim=True)
        conditional_std = conditional.std(dim=dimensions, keepdim=True)
        epsilon = torch.finfo(guided.dtype).eps
        ratio = torch.where(
            guided_std > epsilon,
            conditional_std / guided_std.clamp_min(epsilon),
            torch.ones_like(guided_std),
        )
        rescaled = guided * ratio
        return torch.lerp(guided, rescaled, cfg_rescale)

    def _model_eps(self, model, x_t, timestep, labels, cfg_scale,
                   adaptive_cfg=None, cfg_rescale=0.0):
        """Run the model, applying classifier-free guidance when the scale exceeds 1."""
        scale = self._guidance_scale(timestep, cfg_scale, adaptive_cfg, x_t.ndim)
        if self._is_conditional_only(scale):
            return model(x_t, timestep, labels)

        null_labels = torch.full_like(labels, model.num_classes)
        combined_x = torch.cat([x_t, x_t], dim=0)
        combined_t = torch.cat([timestep, timestep], dim=0)
        combined_labels = torch.cat([labels, null_labels], dim=0)
        eps_cond, eps_uncond = model(combined_x, combined_t, combined_labels).chunk(2, dim=0)
        eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)
        return self.rescale_cfg(eps_guided, eps_cond, cfg_rescale)

    def _model_eps_cfgpp(self, model, x_t, timestep, labels, cfg_scale,
                         adaptive_cfg=None, cfg_rescale=0.0):
        """Return CFG++'s guided and unconditional noise predictions."""
        null_labels = torch.full_like(labels, model.num_classes)
        combined_x = torch.cat([x_t, x_t], dim=0)
        combined_t = torch.cat([timestep, timestep], dim=0)
        combined_labels = torch.cat([labels, null_labels], dim=0)
        eps_cond, eps_uncond = model(combined_x, combined_t, combined_labels).chunk(2, dim=0)
        scale = self._guidance_scale(timestep, cfg_scale, adaptive_cfg, x_t.ndim)
        eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)
        eps_guided = self.rescale_cfg(eps_guided, eps_cond, cfg_rescale)
        return eps_guided, eps_uncond

    @torch.no_grad()
    def p_sample_loop(self, model, shape, labels, cfg_scale=1.0, device='cpu',
                      adaptive_cfg=None, cfg_rescale=0.0):
        """Full ancestral DDPM sampling over all T timesteps."""
        image = torch.randn(shape, device=device)
        for step in reversed(range(self.num_timesteps)):
            timestep = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps = self._model_eps(
                model, image, timestep, labels, cfg_scale,
                adaptive_cfg=adaptive_cfg, cfg_rescale=cfg_rescale,
            )
            x_start = self.predict_x_start_from_noise(image, timestep, eps).clamp(-1, 1)
            mean = (
                extract(self.posterior_mean_coef1, timestep, image.shape) * x_start
                + extract(self.posterior_mean_coef2, timestep, image.shape) * image
            )
            if step > 0:
                noise = torch.randn_like(image)
                log_var = extract(self.posterior_log_variance, timestep, image.shape)
                image = mean + (0.5 * log_var).exp() * noise
            else:
                image = mean
        return image

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, labels, cfg_scale=1.0,
                         num_sampling_steps=50, eta=0.0, device='cpu',
                         adaptive_cfg=None, cfg_rescale=0.0):
        """Deterministic-ish DDIM sampling on a strided subset of timesteps."""
        step_indices = torch.linspace(0, self.num_timesteps - 1, num_sampling_steps, dtype=torch.long)
        step_indices = list(reversed(step_indices.tolist()))

        image = torch.randn(shape, device=device)
        for i, step in enumerate(step_indices):
            timestep = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps = self._model_eps(
                model, image, timestep, labels, cfg_scale,
                adaptive_cfg=adaptive_cfg, cfg_rescale=cfg_rescale,
            )
            x_start = self.predict_x_start_from_noise(image, timestep, eps).clamp(-1, 1)

            alpha_bar = extract(self.alphas_cumprod, timestep, image.shape)
            prev_step = step_indices[i + 1] if i + 1 < len(step_indices) else 0
            prev_timestep = torch.full((shape[0],), prev_step, device=device, dtype=torch.long)
            alpha_bar_prev = extract(self.alphas_cumprod, prev_timestep, image.shape)

            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            )
            direction = torch.sqrt((1 - alpha_bar_prev - sigma ** 2).clamp(min=0)) * eps
            image = torch.sqrt(alpha_bar_prev) * x_start + direction
            if eta > 0 and i + 1 < len(step_indices):
                image = image + sigma * torch.randn_like(image)
        return image

    @torch.no_grad()
    def cfgpp_sample_loop(self, model, shape, labels, cfg_scale=1.0,
                          num_sampling_steps=50, device='cpu', adaptive_cfg=None,
                          cfg_rescale=0.0):
        """DDIM sampling with CFG++: guided x_0 and unconditional noise direction."""
        step_indices = torch.linspace(0, self.num_timesteps - 1, num_sampling_steps, dtype=torch.long)
        step_indices = list(reversed(step_indices.tolist()))

        image = torch.randn(shape, device=device)
        for index, step in enumerate(step_indices):
            timestep = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps_guided, eps_uncond = self._model_eps_cfgpp(
                model, image, timestep, labels, cfg_scale,
                adaptive_cfg=adaptive_cfg, cfg_rescale=cfg_rescale,
            )
            x_start = self.predict_x_start_from_noise(image, timestep, eps_guided).clamp(-1, 1)

            prev_step = step_indices[index + 1] if index + 1 < len(step_indices) else 0
            prev_timestep = torch.full((shape[0],), prev_step, device=device, dtype=torch.long)
            alpha_bar_prev = extract(self.alphas_cumprod, prev_timestep, image.shape)
            direction = torch.sqrt(1 - alpha_bar_prev) * eps_uncond
            image = torch.sqrt(alpha_bar_prev) * x_start + direction
        return image

    @torch.no_grad()
    def sample(self, model, num_images, labels, image_size, image_channel,
               sampler='ddpm', num_sampling_steps=50, cfg_scale=1.0, device='cpu',
               adaptive_cfg=None, cfg_rescale=0.0):
        """Convenience wrapper returning images in [-1, 1]."""
        shape = (num_images, image_channel, image_size, image_size)
        if sampler == "ddim":
            return self.ddim_sample_loop(
                model, shape, labels, cfg_scale, num_sampling_steps,
                device=device, adaptive_cfg=adaptive_cfg, cfg_rescale=cfg_rescale,
            )
        if sampler == "cfgpp":
            return self.cfgpp_sample_loop(
                model, shape, labels, cfg_scale, num_sampling_steps,
                device=device, adaptive_cfg=adaptive_cfg, cfg_rescale=cfg_rescale,
            )
        return self.p_sample_loop(
            model, shape, labels, cfg_scale, device=device,
            adaptive_cfg=adaptive_cfg, cfg_rescale=cfg_rescale,
        )
