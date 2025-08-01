import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    # When the Encoder is initialized, it receives a list of Encoderlyers in the model
    def __init__(self, List_of_Duplicated_FTP_Modules):
        super(Encoder, self).__init__()
        # Convert the list to a Pytorch encapsulated list of modules
        self.List_of_Duplicated_FTP_Modules = nn.ModuleList(
            List_of_Duplicated_FTP_Modules
        )

    def forward(self, x):
        # x: [batch * (channel + 4) * timestep]
        # FTP_Module is the EncoderLayer
        # Because you need to repeat the EncoderLayer, you need to make sure that the input and output shapes of x are the same at each layer
        for FTP_Module in self.List_of_Duplicated_FTP_Modules:
            x = FTP_Module(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, configs, FTP_Local_Global_Fusion, FTP_Channel_Mix, Channel_enhancement
    ):
        super(EncoderLayer, self).__init__()
        self.FTP_Local_Global_Fusion = FTP_Local_Global_Fusion
        self.FTP_Channel_Mix = FTP_Channel_Mix
        self.Channel_enhancement = Channel_enhancement
        self.activation = F.relu if configs.activation == "relu" else F.gelu

        # Layer normalization
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)

        self.dropout = nn.Dropout(configs.dropout)

        # Linear
        self.Linear_embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.Linear_All_Fusion = nn.Linear(configs.d_model * 3, configs.d_model)
        self.Linear_All_Fusion_2 = nn.Linear(configs.d_model * 2, configs.d_model)
        self.Linear_All_Fusion_1 = nn.Linear(configs.d_model, configs.d_model)
        self.Linear_recover_seq_len = nn.Linear(configs.d_model, configs.seq_len)

        # Linear_ablation
        self.Linear_All_Fusion_ablation_0 = nn.Linear(
            configs.d_model * 2, configs.d_model
        )
        self.Linear_All_Fusion_ablation_1 = nn.Linear(
            configs.d_model * 2, configs.d_model
        )
        self.Linear_All_Fusion_ablation_2 = nn.Linear(
            configs.d_model * 2, configs.d_model
        )
        self.Linear_All_Fusion_2_ablation_3 = nn.Linear(
            configs.d_model, configs.d_model
        )
        self.Linear_All_Fusion_ablation_6 = nn.Linear(configs.d_model, configs.d_model)

        # conv
        self.conv1 = nn.Conv1d(
            in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1
        )

        # ablation
        self.ablation_experiment = configs.ablation_experiment
        self.ablation_experiment_type = configs.ablation_experiment_type

    def forward(self, x):
        # ======================ablation code=======================
        if self.ablation_experiment:
            if self.ablation_experiment_type == 0:  # w/o GLF-CI
                # x: [batch * (channel + 4) * timestep]
                x_channel_mix = self.FTP_Channel_Mix(x)  # [batch, channel+4, d_model]
                x_channel_enhancement = self.Channel_enhancement(
                    x
                )  # [batch, channel+4, d_model]

                # The output information of the three modules is fused
                x_tree_module_fusion = torch.cat(
                    (x_channel_mix, x_channel_enhancement), dim=-1
                )  # [batch, channel+4, d_model * 3]
                x_tree_module_fusion = self.Linear_All_Fusion_ablation_0(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]

                x_embedding = self.Linear_embedding(x)  # [batch, channel+4, d_model]

                x_cat_all_fusion = torch.cat(
                    (x_embedding, x_tree_module_fusion), dim=-1
                )  # [batch, channel+4, d_model * 2]
                x_d_model = self.Linear_All_Fusion_2(
                    x_cat_all_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]
                return x  # x: [batch * (channel + 4) * timestep]
            elif self.ablation_experiment_type == 1:  # w/o GLF-CM
                # x: [batch * (channel + 4) * timestep]
                x_local_and_global_fusion = self.FTP_Local_Global_Fusion(
                    x
                )  # [batch, channel + 4, d_model]

                x_channel_enhancement = self.Channel_enhancement(
                    x
                )  # [batch, channel+4, d_model]

                # The output information of the three modules is fused
                x_tree_module_fusion = torch.cat(
                    (x_local_and_global_fusion, x_channel_enhancement), dim=-1
                )  # [batch, channel+4, d_model * 3]
                x_tree_module_fusion = self.Linear_All_Fusion_ablation_1(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]

                x_embedding = self.Linear_embedding(x)  # [batch, channel+4, d_model]

                x_cat_all_fusion = torch.cat(
                    (x_embedding, x_tree_module_fusion), dim=-1
                )  # [batch, channel+4, d_model * 2]
                x_d_model = self.Linear_All_Fusion_2(
                    x_cat_all_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]
            elif self.ablation_experiment_type == 2:  # w/o CE
                # x: [batch * (channel + 4) * timestep]
                x_local_and_global_fusion = self.FTP_Local_Global_Fusion(
                    x
                )  # [batch, channel + 4, d_model]
                x_channel_mix = self.FTP_Channel_Mix(x)  # [batch, channel+4, d_model]

                # The output information of the three modules is fused
                x_tree_module_fusion = torch.cat(
                    (x_local_and_global_fusion, x_channel_mix), dim=-1
                )  # [batch, channel+4, d_model * 3]
                x_tree_module_fusion = self.Linear_All_Fusion_ablation_2(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]

                x_embedding = self.Linear_embedding(x)  # [batch, channel+4, d_model]

                x_cat_all_fusion = torch.cat(
                    (x_embedding, x_tree_module_fusion), dim=-1
                )  # [batch, channel+4, d_model * 2]
                x_d_model = self.Linear_All_Fusion_2(
                    x_cat_all_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]
            elif self.ablation_experiment_type == 3:  # w/o Orinal
                # x: [batch * (channel + 4) * timestep]
                x_local_and_global_fusion = self.FTP_Local_Global_Fusion(
                    x
                )  # [batch, channel + 4, d_model]
                x_channel_mix = self.FTP_Channel_Mix(x)  # [batch, channel+4, d_model]
                x_channel_enhancement = self.Channel_enhancement(
                    x
                )  # [batch, channel+4, d_model]

                # The output information of the three modules is fused
                x_tree_module_fusion = torch.cat(
                    (x_local_and_global_fusion, x_channel_mix, x_channel_enhancement),
                    dim=-1,
                )  # [batch, channel+4, d_model * 3]
                x_tree_module_fusion = self.Linear_All_Fusion(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]

                x_d_model = self.Linear_All_Fusion_2_ablation_3(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]

            elif self.ablation_experiment_type == 4:  # w/o Fusion(Only Orinal)
                # x: [batch * (channel + 4) * timestep]
                x_embedding = self.Linear_embedding(x)  # [batch, channel+4, d_model]
                
                x_d_model = self.Linear_All_Fusion_2_ablation_3(
                    x_embedding
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]

            elif self.ablation_experiment_type == 5:  # OnlyCE
               
                x_channel_enhancement = self.Channel_enhancement(
                    x
                )  # [batch, channel+4, d_model]
               
                x_tree_module_fusion = self.Linear_All_Fusion_ablation_6(
                    x_channel_enhancement
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]

                x_d_model = self.Linear_All_Fusion_1(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]

            elif self.ablation_experiment_type == 6:  # CI + orinal
                # x: [batch * (channel + 4) * timestep]
                x_local_and_global_fusion = self.FTP_Local_Global_Fusion(
                    x
                )  # [batch, channel + 4, d_model]

                x_tree_module_fusion = self.Linear_All_Fusion_ablation_6(
                    x_local_and_global_fusion
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]

                x_embedding = self.Linear_embedding(x)  # [batch, channel+4, d_model]

                x_cat_all_fusion = torch.cat(
                    (x_embedding, x_tree_module_fusion), dim=-1
                )  # [batch, channel+4, d_model * 2]
                x_d_model = self.Linear_All_Fusion_2(
                    x_cat_all_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]

            elif self.ablation_experiment_type == 7:  # CM + orinal
                # x: [batch * (channel + 4) * timestep]
                x_channel_mix = self.FTP_Channel_Mix(x)  # [batch, channel+4, d_model]

                x_tree_module_fusion = self.Linear_All_Fusion_ablation_6(
                    x_channel_mix
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]

                x_embedding = self.Linear_embedding(x)  # [batch, channel+4, d_model]

                x_cat_all_fusion = torch.cat(
                    (x_embedding, x_tree_module_fusion), dim=-1
                )  # [batch, channel+4, d_model * 2]
                x_d_model = self.Linear_All_Fusion_2(
                    x_cat_all_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]

            elif self.ablation_experiment_type == 8:  # CE + orinal
                # x: [batch * (channel + 4) * timestep]
                x_channel_enhancement = self.Channel_enhancement(
                    x
                )  # [batch, channel+4, d_model]

                x_tree_module_fusion = self.Linear_All_Fusion_ablation_6(
                    x_channel_enhancement
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]

                x_embedding = self.Linear_embedding(x)  # [batch, channel+4, d_model]

                x_cat_all_fusion = torch.cat(
                    (x_embedding, x_tree_module_fusion), dim=-1
                )  # [batch, channel+4, d_model * 2]
                x_d_model = self.Linear_All_Fusion_2(
                    x_cat_all_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]

            elif self.ablation_experiment_type == 9:  # Only CI
                # x: [batch * (channel + 4) * timestep]
                x_local_and_global_fusion = self.FTP_Local_Global_Fusion(
                    x
                )  # [batch, channel + 4, d_model]
                x_tree_module_fusion = self.Linear_All_Fusion_ablation_6(
                    x_local_and_global_fusion
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x_d_model = self.Linear_All_Fusion_1(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]

            elif self.ablation_experiment_type == 10:  # Only CM
                # x: [batch * (channel + 4) * timestep]
                x_channel_mix = self.FTP_Channel_Mix(x)  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.Linear_All_Fusion_ablation_6(
                    x_channel_mix
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x_d_model = self.Linear_All_Fusion_1(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]

            elif self.ablation_experiment_type == 11:  # Only CE + CI
                # x: [batch * (channel + 4) * timestep]
                x_local_and_global_fusion = self.FTP_Local_Global_Fusion(
                    x
                )  # [batch, channel + 4, d_model]

                x_channel_enhancement = self.Channel_enhancement(
                    x
                )  # [batch, channel+4, d_model]

                # The output information of the three modules is fused
                x_tree_module_fusion = torch.cat(
                    (x_local_and_global_fusion, x_channel_enhancement), dim=-1
                )  # [batch, channel+4, d_model * 2]
                x_tree_module_fusion = self.Linear_All_Fusion_ablation_1(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]

                x_d_model = self.Linear_All_Fusion_1(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]

            elif self.ablation_experiment_type == 12:  # Only CE + CM
                # x: [batch * (channel + 4) * timestep]
                x_channel_mix = self.FTP_Channel_Mix(x)  # [batch, channel+4, d_model]

                x_channel_enhancement = self.Channel_enhancement(
                    x
                )  # [batch, channel+4, d_model]

                # The output information of the three modules is fused
                x_tree_module_fusion = torch.cat(
                    (x_channel_mix, x_channel_enhancement), dim=-1
                )  # [batch, channel+4, d_model * 2]
                x_tree_module_fusion = self.Linear_All_Fusion_ablation_1(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x_tree_module_fusion = self.activation(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]

                x_d_model = self.Linear_All_Fusion_1(
                    x_tree_module_fusion
                )  # [batch, channel+4, d_model]
                x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
                y = self.norm1(x)  # [batch, channel+4, d_model]
                y = self.dropout(
                    self.activation(self.conv1(y.transpose(-1, 1)))
                )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
                y = self.dropout(
                    self.conv2(y).transpose(-1, 1)
                )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

                x = self.norm2(x + y)  # [batch, channel+4, d_model]
                x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

                return x  # x: [batch * (channel + 4) * timestep]

        # ======================Full code=======================
        else:
            # x: [batch * (channel + 4) * timestep]
            x_local_and_global_fusion = self.FTP_Local_Global_Fusion(
                x
            )  # [batch, channel + 4, d_model]
            x_channel_mix = self.FTP_Channel_Mix(x)  # [batch, channel+4, d_model]
            x_channel_enhancement = self.Channel_enhancement(
                x
            )  # [batch, channel+4, d_model]

            # The output information of the three modules is fused
            x_tree_module_fusion = torch.cat(
                (x_local_and_global_fusion, x_channel_mix, x_channel_enhancement),
                dim=-1,
            )  # [batch, channel+4, d_model * 3]
            x_tree_module_fusion = self.Linear_All_Fusion(
                x_tree_module_fusion
            )  # [batch, channel+4, d_model]
            x_tree_module_fusion = self.activation(
                x_tree_module_fusion
            )  # [batch, channel+4, d_model]

            x_embedding = self.Linear_embedding(x)  # [batch, channel+4, d_model]

            x_cat_all_fusion = torch.cat(
                (x_embedding, x_tree_module_fusion), dim=-1
            )  # [batch, channel+4, d_model * 2]
            x_d_model = self.Linear_All_Fusion_2(
                x_cat_all_fusion
            )  # [batch, channel+4, d_model]
            x = x_d_model + self.dropout(x_d_model)  # [batch, channel+4, d_model]
            y = self.norm1(x)  # [batch, channel+4, d_model]
            y = self.dropout(
                self.activation(self.conv1(y.transpose(-1, 1)))
            )  # [batch,d_ff,channel+4]  # conv1的意思为输入的通道数为d_model，输出的通道数为d_ff，卷积核大小为1，conv1接受的输入的形状为[batch, d_model（通道数），channel+4（序列长度）]
            y = self.dropout(
                self.conv2(y).transpose(-1, 1)
            )  # [batch,d_model,channel+4] => transpose [batch,channel+4,d_model]

            x = self.norm2(x + y)  # [batch, channel+4, d_model]
            x = self.Linear_recover_seq_len(x)  # [batch, channel+4, seq_len]

            return x  # x: [batch * (channel + 4) * timestep]
