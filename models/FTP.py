import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.RevIN import RevIN
from layers.FTP_EncDec import Encoder, EncoderLayer
from layers.Embed import positional_encoding

'''
三个模块类，后续考虑放在layers包中
'''


# Serial Module
class FTP_Local_Global_Fusion_Patching_Level_Layers(nn.Module):
    def __init__(self, configs, local_and_global_adaptive_level):
        super(FTP_Local_Global_Fusion_Patching_Level_Layers, self).__init__()
        """
        Local and global fusion Information Module
        """
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        subtract_last = configs.subtract_last
        context_window = configs.seq_len
        target_window = configs.pred_len
        self.dropout = nn.Dropout(configs.dropout)
        self.seq_len = configs.seq_len
        self.local_and_global_parallel_or_serial = configs.local_and_global_parallel_or_serial
        self.unit_patch_len = configs.unit_patch_len
        # Patching
        self.padding_patch = configs.padding_patch  # padding is used at the end
        self.stride = configs.stride  # patching step length
        self.adaptation_patch_len = local_and_global_adaptive_level * self.unit_patch_len  # The length of patching in each layer，is the number of layers *  self.unit_patch_len
        patch_num = int((context_window - self.adaptation_patch_len) / self.stride + 1)  # The number of patching
        self.is_use_adaptive_stride = configs.is_use_adaptive_stride
        if self.is_use_adaptive_stride:
            self.adaptation_patch_len = local_and_global_adaptive_level * self.unit_patch_len  # The length of patching in each layer，is the number of layers *  self.unit_patch_len
            # self.stride = self.adaptation_patch_len // 2
            self.stride = 1 if self.adaptation_patch_len // 2 == 0 else self.adaptation_patch_len // 2
            patch_num = int((context_window - self.adaptation_patch_len) / self.stride + 1)  # The number of patching
        # PatchTST mentioned in the paper that before Patch, S fillings should be repeated at the last value to the end of the original sequence
        if self.padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            patch_num += 1  # Therefore, the number of patches needs to be increased by 1

        # Positional Encoding
        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=patch_num, d_model=self.d_model)

        # Liner
        self.Linear_Embedding_of_Patching = nn.Linear(self.adaptation_patch_len, self.d_model)  # Embedding of patching
        self.W_G_MLP = nn.Linear(context_window, self.d_model)
        self.W_G_L_CAT = nn.Linear(self.d_model * 2, self.d_model)
        self.flatten = nn.Flatten(start_dim=-2)
        self.Linear_context_window = nn.Linear(self.d_model * patch_num, self.seq_len)

    def forward(self, x):
        # x: [batch * (channel + 4) * timestep]
        # print("x.shape",x.shape)
        x_no_padding = x
        if self.padding_patch == 'end':
            # The last value is filled with stride values
            x = self.padding_patch_layer(x)  # [batch * (channel+4) * (timestep+stride)]
            # x_no_padding = x[:, :, :-self.stride]  # x_no_padding: [batch * channel+4 * timestep]

            z = x.unfold(dimension=-1, size=self.adaptation_patch_len,
                         step=self.stride)  # z: [batch * channel+4 * patch_num * patch_len]
            z = z.permute(0, 1, 3, 2)  # [batch * channel+4 * patch_len * patch_num]
            z = z.permute(0, 1, 3, 2)  # [batch * channel+4 * patch_num * patch_len]
            z_embedding = self.Linear_Embedding_of_Patching(z)  # [batch * channel+4 * patch_num * d_model]

            # Channel independence
            zci = torch.reshape(z_embedding, (
                z_embedding.shape[0] * z_embedding.shape[1], z_embedding.shape[2],
                z_embedding.shape[3]))  # [(batch * channel+4) * patch_num * d_model]

            # Add location coding and dropout
            u = self.dropout(zci + self.W_pos)  # [(batch * channel+4) * patch_num * d_model]

            # Global Information MLP Moudle
            g = self.W_G_MLP(x_no_padding)  # [batch * channel+4 * d_model]

            # Channel independence
            gci = torch.reshape(g, (g.shape[0] * g.shape[1], g.shape[2]))  # [(batch * channel) * d_model]

            # repeat for patch_num
            gci = gci.unsqueeze(1)  # [(batch * channel+4) * 1 * d_model]
            gci = gci.repeat(1, zci.shape[1], 1)  # [(batch * channel+4) * patch_num * d_model]

            # Local and global fusion
            g_l_cat = torch.cat([u, gci], dim=-1)  # [(batch * channel+4) * patch_num * (d_model * 2)]
            g_l_cat = self.activation(self.W_G_L_CAT(g_l_cat))  # [(batch * channel+4) * patch_num * d_model]

            g_l_cat = torch.reshape(g_l_cat,
                                    (-1, g.shape[1], zci.shape[1],
                                     self.d_model))  # [batch * channel+4 * patch_num * d_model]
            g_l_cat = g_l_cat.permute(0, 1, 3, 2)  # [batch * channel+4 * d_model * patch_num]
            g_l_cat = self.flatten(g_l_cat)  # [batch * channel+4 * (d_model * patch_num)]
            g_l_cat = self.Linear_context_window(g_l_cat)  # [batch * channel+4 * seq_len]
            # Make sure the shapes of the inputs and outputs are the same, because they are serial
            g_l_cat = self.dropout(g_l_cat)  # [batch * channel+4 * seq_len]

        return g_l_cat


class FTP_Local_Global_Fusion(nn.Module):
    def __init__(self, configs):
        super(FTP_Local_Global_Fusion, self).__init__()
        self.local_and_global_parallel_or_serial = configs.local_and_global_parallel_or_serial
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        self.local_and_global_adaptive_level = configs.local_and_global_adaptive_level
        self.dropout = nn.Dropout(configs.dropout)
        self.device = torch.device('cuda:{}'.format(configs.gpu))

        # FTP_Local_Global_Fusion_Patching_Level_Layers (Create 4 adaptive layers)
        self.FTP_Local_Global_Fusion_Patching_Level_Layers = nn.ModuleList(
            [FTP_Local_Global_Fusion_Patching_Level_Layers(configs, i).to(self.device)
             for i in
             range(1, self.local_and_global_adaptive_level + 1)])
        # Local_and_Global_Fusion_Information_Embedding_Module
        self.Linear_Local_and_Global_Fusion_Information_Embedding_Module = nn.Linear(configs.seq_len, configs.d_model)

        # For Parallel
        self.Parallel_Local_and_Global_Fusion_Information_Concat = nn.Linear(
            configs.seq_len * self.local_and_global_adaptive_level, configs.d_model)

    def forward(self, x):
        # x: [batch * (channel + 4) * timestep]
        # Iterate through each layer of adaptive Patching modules
        # Make sure the input and output of x are the same shape x: [batch * (channel + 4) * timestep]

        # Serial module
        if self.local_and_global_parallel_or_serial == 'serial':
            for patching_level in self.FTP_Local_Global_Fusion_Patching_Level_Layers:
                x = patching_level(x)  # [batch * (channel + 4) * seq_len]
            x = self.activation(
                self.Linear_Local_and_Global_Fusion_Information_Embedding_Module(x))  # [batch * channel + 4  * d_model]

        # Parallel module
        if self.local_and_global_parallel_or_serial == 'parallel':
            # Store adaptive output
            # adapt_output = []
            sum_output = None
            for patching_level in self.FTP_Local_Global_Fusion_Patching_Level_Layers:
                # adapt_output.append(patching_level(x))  # [batch * (channel + 4) * seq_len]
                output = patching_level(x)  # [batch * (channel + 4) * seq_len]
                if sum_output is None:
                    sum_output = output
                else:
                    sum_output = sum_output + output
            # sum_output = sum_output + x
            x = self.activation(
                self.Linear_Local_and_Global_Fusion_Information_Embedding_Module(
                    sum_output))  # [batch * channel + 4  * d_model]

            # Concatenate all output on the last dimension
            # x_concat = torch.cat(adapt_output, dim=-1)  # [batch * (channel + 4) * (seq_len * local_and_global_adaptive_level)]
            # x = self.Parallel_Local_and_Global_Fusion_Information_Concat(x_concat)  # [batch * (channel + 4) * d_model]
            # x = self.activation(x)  # [batch * (channel + 4) * d_model]
        return x


class FTP_Channel_Mix_Patching_Level_Layers(nn.Module):
    def __init__(self, configs, channel_mix_information_adaptive_level):
        super(FTP_Channel_Mix_Patching_Level_Layers, self).__init__()
        """
        FTP Channel Mix Module
        """
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.dropout = nn.Dropout(configs.dropout)
        context_window = configs.seq_len
        target_window = configs.pred_len
        # Think about it, there should be no need for +4 operation, because you can directly judge the total number of input channels from the input shape of x, and then consider modifying
        self.time_encoder = configs.time_encoder

        if self.time_encoder:
            self.channel_num = configs.c_in + 4
        else:
            self.channel_num = configs.c_in

        self.seq_len = configs.seq_len
        self.unit_patch_len = configs.unit_patch_len
        self.is_use_adaptive_stride = configs.is_use_adaptive_stride
        # Patching
        self.padding_patch = configs.padding_patch
        self.adaptation_patch_len = channel_mix_information_adaptive_level * self.unit_patch_len
        self.stride = configs.stride
        patch_num = int((context_window - self.adaptation_patch_len) / self.stride + 1)
        self.channel_mix_information_parallel_or_serial = configs.channel_mix_information_parallel_or_serial
        self.is_use_adaptive_stride = configs.is_use_adaptive_stride
        if self.is_use_adaptive_stride:
            self.adaptation_patch_len = channel_mix_information_adaptive_level * self.unit_patch_len  # The length of patching in each layer，is the number of layers *  self.unit_patch_len
            # self.stride = self.adaptation_patch_len // 2
            self.stride = 1 if self.adaptation_patch_len // 2 == 0 else self.adaptation_patch_len // 2
            patch_num = int((context_window - self.adaptation_patch_len) / self.stride + 1)  # The number of patching

        # PatchTST mentioned in the paper that before Patch, S fillings should be repeated at the last value to the end of the original sequence
        if self.padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            patch_num += 1  # Therefore, the number of patches needs to be increased by 1

        # Positional Encoding
        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=patch_num, d_model=self.d_model)

        # Liner
        self.Linear_Embedding_of_Patching = nn.Linear(self.adaptation_patch_len * self.channel_num,
                                                      self.d_model)  # Embedding of patching

        self.W_M_MLP = nn.Linear(self.channel_num * context_window, self.d_model)
        self.W_G_L_M_CAT = nn.Linear(self.d_model * 2, self.d_model)
        self.Linear_recover_channel_patch_nen = nn.Linear(self.d_model, self.channel_num * self.adaptation_patch_len)
        self.flatten = nn.Flatten(start_dim=-2)
        self.Linear_context_window = nn.Linear(self.adaptation_patch_len * patch_num, self.seq_len)

    def forward(self, x):
        # x: [batch * channel+4 * timestep]
        x_no_padding = x
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)

            z = x.unfold(dimension=-1, size=self.adaptation_patch_len,
                         step=self.stride)  # z: [batch * channel+4 * patch_num * patch_len]
            z = z.permute(0, 1, 3, 2)  # [batch * channel+4 * patch_len * patch_num] (16 * 325 * 16 * 12)
            # z = z.permute(0, 1, 3, 2)  # [batch * channel+4 * patch_num * patch_len] (16 * 325 * 12 * 16)
            # z_embedding = self.Linear_Embedding_of_Patching(z)  # [batch * channel+4 * patch_num * d_model]

            # Channel Mixing
            zcm = torch.reshape(z, (
                z.shape[0], z.shape[1] * z.shape[2], z.shape[3]))  # [batch * (channel+4 * patch_len) * patch_num]
            zcm = zcm.permute(0, 2, 1)  # [batch * patch_num * (channel+4 * patch_len)]
            zcm_embedding = self.Linear_Embedding_of_Patching(zcm)  # [batch * patch_num * d_model]

            # Add location coding and dropout
            u = self.dropout(zcm_embedding + self.W_pos)  # [batch * patch_num * d_model]

            # Channel mix information MLP Moudle
            g = torch.reshape(x_no_padding, (
                x_no_padding.shape[0],
                x_no_padding.shape[1] * x_no_padding.shape[2]))  # [batch * (channel+4 * timestep)]
            gcm = self.W_M_MLP(g)  # [batch * (d_model)]

            # repeat for patch_num
            gcm = gcm.unsqueeze(1)  # [batch * 1 * (d_model)]
            gcm = gcm.repeat(1, zcm.shape[1], 1)  # [batch * patch_num * (d_model)]

            # Local and Global mixer information fusion
            g_l_cat = torch.cat([u, gcm], dim=-1)  # [batch * patch_num * (d_model * 2)]
            g_l_cat = self.activation(self.W_G_L_M_CAT(g_l_cat))  # [batch * patch_num * d_model]
            g_l_cat = self.Linear_recover_channel_patch_nen(g_l_cat)  # [batch * patch_num * (channel+4 * patch_len)]
            g_l_cat = g_l_cat.reshape(g_l_cat.shape[0], g_l_cat.shape[1], self.channel_num,
                                      self.adaptation_patch_len)  # [batch * patch_num * channel+4 * patch_len]
            g_l_cat = g_l_cat.permute(0, 2, 1, 3)  # [batch * channel+4 * patch_num * patch_len]
            g_l_cat = self.flatten(g_l_cat)  # [batch * channel *  (patch_num * patch_len)]
            g_l_cat = self.Linear_context_window(g_l_cat)  # [batch * channel+4 * seq_len]
            g_l_cat = self.dropout(g_l_cat)  # [batch * channel+4 * seq_len]

        return g_l_cat


class FTP_Channel_Mix(nn.Module):
    def __init__(self, configs):
        super(FTP_Channel_Mix, self).__init__()
        self.channel_mix_information_parallel_or_serial = configs.channel_mix_information_parallel_or_serial
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.device = torch.device('cuda:{}'.format(configs.gpu))
        self.channel_mix_information_adaptive_level = configs.channel_mix_information_adaptive_level
        # FTP_Channel_Mix_Patching_Level_Layers (Create 4 adaptive layers)
        self.FTP_Channel_Mix_Patching_Level_Layers = nn.ModuleList(
            [FTP_Channel_Mix_Patching_Level_Layers(configs, i).to(self.device)
             for i in
             range(1, self.channel_mix_information_adaptive_level + 1)])
        self.Linear_Channel_Mix_Information_Embedding_Module = nn.Linear(configs.seq_len, configs.d_model)
        # For Parallel
        self.Parallel_FTP_Channel_Mix_Concat = nn.Linear(
            configs.seq_len * self.channel_mix_information_adaptive_level, configs.d_model)

    def forward(self, x):
        # x: [batch * channel+4 * timestep]
        # Iterate through each layer of adaptive Patching modules

        # Serial module
        if self.channel_mix_information_parallel_or_serial == 'serial':
            for channel_mix_level in self.FTP_Channel_Mix_Patching_Level_Layers:
                x = channel_mix_level(x)  # [batch * channel+4 * seq_len]
            x = self.activation(
                self.Linear_Channel_Mix_Information_Embedding_Module(x))  # [batch * channel+4 * d_model]
        # Parallel module
        if self.channel_mix_information_parallel_or_serial == 'parallel':
            # Store adaptive output
            # adapt_output = []
            sum_output = None
            for patching_level in self.FTP_Channel_Mix_Patching_Level_Layers:
                output = patching_level(x)  # [batch * (channel + 4) * seq_len]
                if sum_output is None:
                    sum_output = output
                else:
                    sum_output = sum_output + output
            # Residual error
            # sum_output = sum_output  +  x

            x = self.activation(
                self.Linear_Channel_Mix_Information_Embedding_Module(sum_output))  # [batch * channel+4 * d_model]
            # Concatenate all output on the last dimension
            # x_concat = torch.cat(adapt_output,
            #                      dim=-1)  # [batch * (channel + 4) * (seq_len * local_and_global_adaptive_level)]
            # x = self.Parallel_FTP_Channel_Mix_Concat(x_concat)  # [batch * (channel + 4) * d_model]
            # x = self.activation(x)  # [batch * (channel + 4) * d_model]
        return x


class Channel_enhancement(nn.Module):
    """
    Channel enhancement module
    """

    def __init__(self, configs):
        super(Channel_enhancement, self).__init__()
        self.Linear_Embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.dropout = nn.Dropout(configs.dropout)
        self.Linear_output = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        # x: [batch * channel+4 * timestep]
        x_embedding = self.Linear_Embedding(x)  # [batch * channel * d_model]
        x_feature_softmax = F.softmax(x_embedding, dim=-1)  # [batch * channel * d_model]
        feature_gated_attention = x_embedding * x_feature_softmax  # [batch * channel * d_model]

        # Residual error
        x_attn = feature_gated_attention + x_embedding  # [batch * channel * d_model]
        x_attn = self.dropout(x_attn)
        ratio = x_attn.permute(0, 2, 1)  # [batch * d_model * channel]

        # softmax
        ratio = F.softmax(ratio, dim=-1)  # [batch * d_model * channel]
        ratio = ratio.reshape(-1, x_embedding.shape[1])  # [(batch * d_model), channel]

        # sampling
        indices = torch.multinomial(ratio, 1)  # [(batch * d_model), 1]
        indices = indices.reshape(x_embedding.shape[0], -1, 1)  # [batch, d_model, 1]
        indices = indices.permute(0, 2, 1)  # [batch, 1, d_model]
        x_attn = torch.gather(x_attn, 1, indices)  # [batch, 1, d_model]
        x_attn = x_attn.repeat(1, x_embedding.shape[1], 1)  # [batch, channel, d_model]
        x_attn = self.activation(self.Linear_output(x_attn))  # [batch, channel, d_model]
        return x_attn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm
        self.time_encoder = configs.time_encoder
        c_in = configs.enc_in
        affine = configs.affine
        subtract_last = configs.subtract_last
        context_window = configs.seq_len
        target_window = configs.pred_len

        # RevIN
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs,
                    FTP_Local_Global_Fusion(configs),
                    FTP_Channel_Mix(configs),
                    Channel_enhancement(configs),
                ) for l in range(configs.e_layers)
            ],
        )

        # Decoder
        self.to_feature = nn.Linear(self.seq_len, configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # x: [batch * timestep * channel
        # x_mark_enc: [batch * timestep * 4]
        N = x.shape[-1]
        x = x.permute(0, 2, 1)  # [batch * channel * timestep]
        if self.revin:
            x = x.permute(0, 2, 1)  # [batch * timestep * channel]
            x = self.revin_layer(x, 'norm')  # [batch * timestep * channel]
            x = x.permute(0, 2, 1)  # [batch * channel * timestep]

        # cat x and x_mark_enc
        if self.time_encoder:
            if x_mark_enc is not None:  # 检查 x_mark_enc 是否为 None
                x = torch.cat([x, x_mark_enc.permute(0, 2, 1)], dim=1)  # [batch * (channel + 4) * timestep]

        x = self.encoder(x)  # [batch * (channel + 4) * timestep]
        x = self.to_feature(x)  # [batch * (channel + 4) * d_model]

        x = self.projection(x)  # [batch * (channel + 4) * pred_len]

        if self.revin:
            x = x.permute(0, 2, 1)  # [batch * pred_len * (channel + 4)]
            x = x[:, :, :N]
            x = self.revin_layer(x, 'denorm')  # [batch * pred_len * channel]
            x = x.permute(0, 2, 1)  # [batch * channel * pred_len]
        x = x.permute(0, 2, 1)  # [batch * pred_len * channel]

        return x
