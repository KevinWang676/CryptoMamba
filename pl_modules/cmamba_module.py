import torch.nn as nn
from models.cmamba import CMamba
from .base_module import BaseModule
    

class CryptoMambaModule(BaseModule):

    def __init__(
        self,
        num_features=5,
        hidden_dims=[14, 1],
        norm_layer=nn.LayerNorm,
        d_conv=4,
        layer_density=1,
        expand=2,
        mlp_ratio=0,
        drop=0.0,
        num_classes=None,
        d_states=16,
        use_checkpoint=False,
        lr=0.0002,
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=0.0,
        logger_type=None,
        window_size=14,
        y_key='Close',
        optimizer='adam',
        mode='default',
        loss='rmse',
        task='regression',
        **kwargs
    ):
        super().__init__(lr=lr,
                         lr_step_size=lr_step_size,
                         lr_gamma=lr_gamma,
                         weight_decay=weight_decay,
                         logger_type=logger_type,
                         y_key=y_key,
                         optimizer=optimizer,
                         mode=mode,
                         window_size=window_size,
                         loss=loss,
                         task=task,
                         )
        assert window_size == hidden_dims[0]

        self.model = CMamba(
            num_features=num_features,
            hidden_dims=hidden_dims,
            norm_layer=norm_layer,
            d_conv=d_conv,
            layer_density=layer_density,
            expand=expand, 
            mlp_ratio=mlp_ratio, 
            drop=drop, 
            num_classes=num_classes,
            d_states=d_states,
            use_checkpoint=use_checkpoint,
            **kwargs
        )
