import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer,_attenion_over_fine_grained_sim_matrix,Logit_text_video
from model.clip_model import Extract,Extract1,TransformerClip,_mean_pooling_for_similarity_visual
import torch.nn.functional as F

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)

        self.extract = Extract()
        self.extract1 = Extract1()
        self._attenion_over_fine_grained_sim_matrix = _attenion_over_fine_grained_sim_matrix(config)
        self.Logit_text_video = Logit_text_video(config)
        self.frame_position_embeddings = nn.Embedding(77, 512)
        self.transformerClip = TransformerClip(width=512, layers=4,heads=8)
        self.clip.logit_scale1 = torch.tensor(2.303, device='cuda:0', requires_grad=True)

    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        word = data['new_sentence']
        video_data = data['video']
        video_mask = data['video_mask']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
        else:
            # text_features = self.clip.encode_text(text_data)
            text_features,hidden = self.clip.encode_text(text_data, return_hidden = True)
            noun_features = self.clip.encode_text(word, return_hidden = False)
            video_features = self.clip.encode_image(video_data)
   
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        "phrase"
        phrase_feat = self.extract(hidden)
        "frame-enhance"
        video_features_clip = self.extract1(video_features)

        video_features_pooled = self.pool_frames(text_features, video_features)
        "noun-frame"
        # video_features_pooled_noun = self.pool_frames(noun_features, video_features)
        """video"""
        video_features1 = video_features
        seq_length = video_features1.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=video_features1.device)
        position_ids = position_ids.unsqueeze(0).expand(video_features1.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        video_features1 = video_features1 + frame_position_embeddings

        extended_video_mask = (1.0 - video_mask) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        video_features1 = video_features1.permute(1, 0, 2)  # NLD -> LND
        video_features1 = self.transformerClip(video_features1, extended_video_mask)
        video_features1 = video_features1.permute(1, 0, 2)  # LND -> NLD
        video_features1 = video_features1 + video_features
        video_features1 = video_features1 / video_features1.norm(dim=-1,keepdim=True)
        video_mean = _mean_pooling_for_similarity_visual(video_features1,video_mask)        

        video_mean = video_mean / video_mean.norm(dim=-1,keepdim=True)
        noun_features = noun_features / noun_features.norm(dim=-1,keepdim=True)
        video_nouns_logits =  self.Logit_text_video(noun_features,video_mean)

        
        "phrase-video"
        phrase_feat = F.normalize(phrase_feat, p=2, dim=-1)
        video_features_clip = F.normalize(video_features_clip, p=2, dim=-1)
        video_features = F.normalize(video_features, p=2, dim=-1)
        word_frames_logits =  self._attenion_over_fine_grained_sim_matrix(phrase_feat,video_features)
        # logit_scale = self.clip.logit_scale1.exp()
        # word_frames_logits = word_frames_logits * logit_scale
        logit = (word_frames_logits + video_nouns_logits) * 0.5
        # logit = video_nouns_logits

        # 源代码！
        # if return_all_frames:
        #     return text_features, video_features, video_features_pooled

        # return text_features, video_features_pooled

        # 改进代码！
        if return_all_frames:
            return {
                'text_embeds': text_features,
                'phrase_feat': phrase_feat,
                'video_features' : video_features,
                'video_features_pooled':video_features_pooled,
                # 'video_features_pooled_noun':video_features_pooled_noun,
                'video_mean' : video_mean,
                'noun_features':noun_features,
                'video_features_clip':video_features_clip,
                'logits' : logit

            }

        return {
        'text_embeds': text_features,
        'noun_features':noun_features,
        'video_features_pooled': video_features_pooled,

        # 'video_features_pooled_noun':video_features_pooled_noun,
        'logits' : logit,
         }