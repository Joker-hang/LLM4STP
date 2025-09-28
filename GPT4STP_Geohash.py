import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel, GPT2TokenizerFast
from einops import rearrange
# from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from models.transformer import TransformerModel

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class GPT4STP(nn.Module):

    def __init__(self, configs, device):
        super(GPT4STP, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.prompt_len = configs.prompt_len
        self.prompt_len_ = 16
        self.head_nf = self.patch_num * self.d_model

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                      output_hidden_states=True, attn_implementation="eager")  # loads a pretrained GPT-2 base model
                self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2',
                                                               trust_remote_code=True, local_files_only=True)
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            # print("gpt2 = {}".format(self.gpt2))

        # 设置 pad_token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        self.in_layer = nn.Linear(self.patch_size, configs.d_model)

        self.time_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1)
        )

        emsize = 32  # embedding dimension
        nhid = 512  # the dimension of the feedforward network model in TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 8  # the number of heads in the multihead-attention models
        self.spatial_encoder = TransformerModel(emsize, nhead, nhid, nlayers, configs.dropout).to(device=device)
        self.input_embedding_layer_spatial = nn.Linear(2, 32).to(device=device)
        self.spatial_embedding = nn.Linear(32, 2).to(device=device)

        self.geohash_encoder = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        ).to(device=device)
        
        self.input_embedding_layer_geohash = nn.Linear(configs.geohash_size * 2, 32).to(device=device)
        self.geohash_embeddding = nn.Linear(32, configs.d_model).to(device=device)

        # 卷积网络提取高斯分布图特征
        self.gaussian_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        ).to(device=device)
        self.gaussian_fc = nn.Linear(32, configs.d_model).to(device=device)

        self.time_feature_extractor.to(device=device)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, self.pred_len)

        # self.prompt_projection = nn.Linear(self.prompt_len, self.prompt_len_)
        # self.output_projection = FlattenHead(self.head_nf, self.pred_len, head_dropout=configs.dropout)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

        self.cnt = 0
        self.description = ("This dataset is a ship AIS trajectory dataset collected every 10 seconds "
                            "in Chengshanjiao water area, which includes the trajectories of all ships "
                            "in the water area at the same time.")
        self.task_description = (f"predict the position information of moving ships for the next {str(self.pred_len)} "
                                 f"time steps ({str(self.pred_len * 10)} s) based on the given position information "
                                 f"from the previous {str(self.seq_len)} time steps ({str(self.seq_len * 10)}s).")
        self.description_nba = ("This is a dataset of the movement trajectories of NBA players on the court. "
                                "The data at each moment includes the trajectory of 10 players and one game ball "
                                "on the field. The interval between adjacent time steps is 0.4s.")
        self.task_description_nba = (f"Predict the position information of 10 players on the field and the game ball for "
                                     f"the next {str(self.pred_len)} time steps ({str(self.pred_len * 0.4)} s) based on "
                                     f"the given position information from the previous {str(self.seq_len)} "
                                     f"time steps ({str(self.seq_len * 0.4)}s).")

    def forward(self, x, total_gaussian, geohash_code):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        input_embedding_spatial = self.input_embedding_layer_spatial(x)
        spatial_embedding = self.spatial_encoder(input_embedding_spatial.permute(1, 0, 2), mask=None)
        spatial_embedding = self.spatial_embedding(spatial_embedding).permute(1, 0, 2)

        x = x + spatial_embedding

        # Generate statistics for prompt
        min_values = torch.min(x, dim=1)[0]
        max_values = torch.max(x, dim=1)[0]
        medians = torch.median(x, dim=1).values
        velocities_ = torch.diff(x, dim=1)
        velocities = torch.diff(x, dim=1).mean(dim=1)
        accelerations = torch.diff(velocities_, dim=1).mean(dim=1)

        prompt = []
        for b in range(x.shape[0]):
            min_values_str = str("%.8f" % min_values[b].tolist()[0])
            max_values_str = str("%.8f" % max_values[b].tolist()[0])
            median_values_str = str("%.8f" % medians[b].tolist()[0])
            velocities_str = str("%.8f" % velocities[b].tolist()[0])
            acceleration_str = str("%.8f" % accelerations[b].tolist()[0])

            prompts = [
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: {self.task_description}"
                f"Statistics for the input trajectory: "
                f"min position {min_values_str}, "
                f"max position {max_values_str}, "
                f"median position {median_values_str}, "
                f"Average velocity {velocities_str}, "
                f"Average acceleration {acceleration_str}."
                f"Acceleration trend: {'Accelerating' if accelerations[b].mean() > 0 else 'Decelerating'}. "
                f"Prediction Focus: Estimate future positions considering the current velocity and acceleration trend."
                f"<|end_prompt|>"
            ]
            prompt.append(prompts[0])

        # Tokenize and encode prompt
        prompt = self.tokenizer(prompt, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=self.prompt_len).input_ids
        prompt = torch.mean(prompt.float(), dim=1)
        # (batch, prompt_token, dim)
        prompt_embeddings = self.gpt2.get_input_embeddings()(prompt.long().to(x.device)).unsqueeze(1)
        prompt_embeddings = prompt_embeddings.repeat(2, 1, 1)

        # [B, M, L]
        x = rearrange(x, 'b l m -> b m l')
        # Extract patches
        # [B, M, N, P]
        x = self.padding_patch_layer(x)
        # 将输入tensor沿着指定维度分割成多个切片
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        patch_embeddings = self.in_layer(x)

        # 在 patch_embeddings 上进行时间维度特征提取
        patch_embeddings = rearrange(patch_embeddings, '(b m) n d -> (b n) d m', b=B)
        patch_embeddings = self.time_feature_extractor(patch_embeddings)  # 时间特征提取
        patch_embeddings = rearrange(patch_embeddings, '(b n) d m -> (b m) n d', b=B)

        # 提取高斯分布图特征
        gaussian_features = self.gaussian_feature_extractor(total_gaussian.unsqueeze(1))  # [num_vehicles, 32, 1, 1]
        gaussian_features = gaussian_features.view(gaussian_features.size(0), -1)  # [num_vehicles, 32]
        gaussian_features = self.gaussian_fc(gaussian_features)  # [num_vehicles, d_model]

        # 将高斯分布图特征扩展为与 batch 维度一致
        gaussian_features = gaussian_features.unsqueeze(1).repeat(2, 1, 1)  # [B * num_vehicles, d_model]

        input_embedding_geohash = self.input_embedding_layer_geohash(geohash_code)
        geohash_embedding = self.geohash_encoder(input_embedding_geohash)
        
        geohash_embedding = self.geohash_embeddding(geohash_embedding)
        geohash_embedding = geohash_embedding.repeat(2, 1, 1)
        # Combine prompt embeddings with patch embeddings
        combined_embeddings = torch.cat([prompt_embeddings, gaussian_features, geohash_embedding, patch_embeddings], dim=1)

        outputs = self.gpt2(inputs_embeds=combined_embeddings).last_hidden_state[:, 3:, :]

        # outputs = rearrange(outputs, '(b m) n p -> b m n p', m=M)

        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
