import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

class MultiHeadedAttention(nn.Module):
    def __init__(self, config: Config):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.cross_attn = MultiHeadedAttention(config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out


class _attenion_over_fine_grained_sim_matrix(nn.Module):
    def __init__(self, config: Config):
        super(_attenion_over_fine_grained_sim_matrix, self).__init__()
        self.local_mat_weight = nn.parameter.Parameter(torch.eye(512), requires_grad=True)
        self.local_mat_weight1 = nn.parameter.Parameter(torch.eye(512), requires_grad=True)
        self.word_mat_weight = nn.parameter.Parameter(torch.eye(14), requires_grad=True)
        self.frame_mat_weight = nn.parameter.Parameter(torch.eye(12), requires_grad=True)
        self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(12), requires_grad=True)
        self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(14), requires_grad=True)

    def forward(self,hidden,video_features1):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        hidden = hidden.to(device)
        video_features1 = video_features1.to(device)
        bs_video, num_frames, dim_video = video_features1.shape
        bs_text, num_words, dim_text = hidden.shape
        # fine_grained_sim_scores = torch.matmul(torch.matmul(hidden.view(-1, dim_text), self.local_mat_weight), torch.matmul(video_features1.view(-1, dim_video),self.local_mat_weight1).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]
        fine_grained_sim_scores = torch.matmul(torch.matmul(hidden.reshape(-1, dim_text), self.local_mat_weight), torch.matmul(video_features1.reshape(-1, dim_video),self.local_mat_weight1).t()).reshape(bs_text, num_words, bs_video, num_frames)
        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        frame_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_video]

        sent2frame_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.frame_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        video2word_logits = torch.sum(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        return (sent2frame_logits + video2word_logits) / 2


class Logit_text_video(nn.Module):
    def __init__(self, config: Config):
        super(Logit_text_video, self).__init__()
        self.global_mat_weight = nn.parameter.Parameter(torch.eye(512), requires_grad=True)
        self.video_mat_weight = nn.parameter.Parameter(torch.eye(512), requires_grad=True)

    def forward(self,text,video_mean):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        text = text.to(device)
        video_mean = video_mean.to(device)

        video_tetx_logits =  torch.matmul(torch.matmul(text, self.global_mat_weight), torch.matmul(video_mean,self.video_mat_weight).t())
        return video_tetx_logits