
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from vica.data.vica_datamanager import VICADataManagerConfig
from vica.vica import VICAModelConfig
from vica.vica_pipeline import VICAPipelineConfig
from vica.vica_trainer import VICATrainerConfig

vica_method = MethodSpecification(
    config=VICATrainerConfig(
        method_name="vica",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=15000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=VICAPipelineConfig(
            datamanager=VICADataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                patch_size=32,
            ),
            model=VICAModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
            ip2p_use_full_precision=True
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF primary method: uses LPIPS, IP2P at full precision",
)

vica_method_small = MethodSpecification(
    config=VICATrainerConfig(
        method_name="vica-small",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=VICAPipelineConfig(
            datamanager=VICADataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                patch_size=32,
            ),
            model=VICAModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
            ip2p_use_full_precision=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF small method, uses LPIPs, IP2P at half precision",
)

vica_method_tiny = MethodSpecification(
    config=VICATrainerConfig(
        method_name="vica-tiny",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=VICAPipelineConfig(
            datamanager=VICADataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=1,
            ),
            model=VICAModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=False,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
            ip2p_use_full_precision=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF tiny method, does not use LPIPs, IP2P at half precision",
)