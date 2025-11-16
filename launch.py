import argparse
import contextlib
import importlib
import logging
import os
import sys
import time
import traceback

# If you want to run the code without internet access, uncomment the following lines
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["DIFFUSERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['PATH'] += ':/envs/dreamdpo/bin/'
os.environ['PATH'] += ':/usr/local/cuda-11.8/bin/'


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def load_custom_module(module_path):
    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
    try:
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(
                module_name, module_path
            )
        else:
            module_spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(module_path, "__init__.py")
            )

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(traceback.format_exc())
        print(f"Cannot import {module_path} module for custom nodes:", e)
        return False


def load_custom_modules():
    node_paths = ["custom"]
    node_import_times = []
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if (
                os.path.isfile(module_path)
                and os.path.splitext(module_path)[1] != ".py"
            ):
                continue
            if module_path.endswith("_disabled"):
                continue
            time_before = time.perf_counter()
            success = load_custom_module(module_path)
            node_import_times.append(
                (time.perf_counter() - time_before, module_path, success)
            )

    if len(node_import_times) > 0:
        print("\nImport times for custom modules:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()


def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    torch.set_float32_matmul_precision("medium")

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.misc import get_rank
    from threestudio.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    load_custom_modules()

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    if args.gradio:
        fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs"))
        fh.setLevel(logging.INFO)
        if args.verbose:
            fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(
                os.path.join(cfg.trial_dir, "code"), use_version=False
            ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        if args.gradio:
            callbacks += [
                ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))
            ]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
                       TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
                       CSVLogger(cfg.trial_dir, name="csv_logs"),
                   ] + system.get_loggers()
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()

    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

    def run_imagereward_eval(system, dm, cfg):
        """Render test views and evaluate them with ImageReward, then save a JSON file.

        This follows the paper's protocol:
        "For each 3D asset, we uniformly render 120 RGB images from different viewpoints
        with the existing code base. Afterward, the ImageReward score is computed from
        the multi-view renderings and averaged for each prompt."
        """
        # Only run on rank 0 to avoid duplicate work
        if get_rank() != 0:
            return

        try:
            from threestudio.reward.imagereward import ImageRewardScore  # type: ignore
        except Exception as e:
            logging.warning(f"[ImageReward] Failed to import ImageRewardScore: {e}")
            return

        # Try to obtain the text prompt from config; fall back to empty string
        prompt = ""
        try:
            prompt = cfg.system.prompt_processor.prompt  # type: ignore[attr-defined]
        except Exception:
            pass

        if prompt is None:
            prompt = ""

        logging.info(f"[ImageReward] Evaluating with prompt: {prompt!r}")

        # Prepare test dataloader (typically uses n_test_views, e.g. 120 views)
        dm.setup("test")
        test_loader = dm.test_dataloader()

        # Initialize reward model (uses default weights path in Config)
        reward = ImageRewardScore({})

        all_scores = []

        system.eval()
        import torch as _torch
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        system = system.to(device)

        with _torch.no_grad():
            for batch in test_loader:
                batch_on_device = {
                    k: (v.to(system.device) if isinstance(v, _torch.Tensor) else v)
                    for k, v in batch.items()
                }

                # Forward pass to get rendered RGB images
                out = system(batch_on_device)
                if "comp_rgb" not in out:
                    logging.warning(
                        "[ImageReward] 'comp_rgb' not found in system output; skipping batch."
                    )
                    continue

                images = out["comp_rgb"]  # [B, H, W, 3] or [B, 3, H, W]
                if images.ndim != 4:
                    logging.warning(
                        f"[ImageReward] Expected comp_rgb to be 4D, got shape {tuple(images.shape)}; skipping batch."
                    )
                    continue

                # Ensure [B, 3, H, W]
                if images.shape[1] != 3 and images.shape[-1] == 3:
                    images = images.permute(0, 3, 1, 2)

                # ImageRewardScore expects images in [0, 1]
                images = _torch.clamp(images, 0.0, 1.0)

                scores = reward(images, prompt)
                if isinstance(scores, _torch.Tensor):
                    all_scores.append(scores.detach().cpu().view(-1))

        if len(all_scores) == 0:
            logging.warning("[ImageReward] No scores collected; nothing to save.")
            return

        all_scores_tensor = _torch.cat(all_scores, dim=0)
        mean_score = float(all_scores_tensor.mean().item())

        import json

        result = {
            "prompt": prompt,
            "num_images": int(all_scores_tensor.shape[0]),
            "image_reward_mean": mean_score,
            "image_reward_all": all_scores_tensor.tolist(),
        }

        os.makedirs(cfg.trial_dir, exist_ok=True)
        out_path = os.path.join(cfg.trial_dir, "imagereward_eval.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        logging.info(
            f"[ImageReward] Mean ImageReward score over {int(all_scores_tensor.shape[0])} views: {mean_score:.4f}"
        )
        logging.info(f"[ImageReward] Saved ImageReward evaluation to {out_path}")

    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
        if args.gradio:
            # also export assets if in gradio mode
            trainer.predict(system, datamodule=dm)
    elif args.validate:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)

    # Optional ImageReward evaluation on the rendered test views
    if getattr(args, "eval_imagereward", False):
        run_imagereward_eval(system, dm, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
             "1,2 means use the 2nd and 3rd available GPU. "
             "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
             "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    parser.add_argument(
        "--eval-imagereward",
        action="store_true",
        help="if true, run ImageReward evaluation on the test views after the main run",
    )

    args, extras = parser.parse_known_args()

    if args.gradio:
        # FIXME: no effect, stdout is not captured
        with contextlib.redirect_stdout(sys.stderr):
            main(args, extras)
    else:
        main(args, extras)
