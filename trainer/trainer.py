from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id
from tqdm import tqdm
import torch.nn.functional as F

class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        # eval_steps = np.linspace(0, num_steps-1, 2600, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):
            # then assume we must tokenize the input, e.g. its a string
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)
            if self.tokenizer is not None:
                data['new_sentence'] = self.tokenizer(data['new_sentence'], return_tensors='pt', padding=True,
                                              truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
            
            if isinstance(data['new_sentence'], torch.Tensor):
                data['new_sentence'] = data['new_sentence'].to(self.device)
            else:
                data['new_sentence'] = {key: val.to(self.device) for key, val in data['new_sentence'].items()}

            
            data['video'] = data['video'].to(self.device)

            # text_embeds, video_embeds_pooled = self.model(data)
            result = self.model(data)
            text_embeds = result['text_embeds']
            # noun_features = result['noun_features']
            video_embeds_pooled = result['video_features_pooled']
            # video_features_pooled_noun = result['video_features_pooled_noun']
            logits =result['logits']


            output1 = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)

            # output2 = sim_matrix_training(noun_features, video_features_pooled_noun, self.pooling_type)


            output = (output1 + logits ) * 0.5
            
            loss = self.loss(output, self.model.clip.logit_scale)
            # loss2 = self.loss(logits, self.model.clip.logit_scale)
            # loss = (loss1 + 0.5 * loss2)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()

                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']
                    # self._save_checkpoint(epoch, save_best=True)

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']
                    self._save_checkpoint(epoch, save_best=True)

                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        phrase_embed_arr = []
        farm_embed_arr = []
        noun_embed_arr = []
        video_mean__embed_arr = []

        all_vid_ids = []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                if self.tokenizer is not None:
                    data['new_sentence'] = self.tokenizer(data['new_sentence'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['new_sentence'], torch.Tensor):
                    data['new_sentence'] = data['new_sentence'].to(self.device)
                else:
                    data['new_sentence'] = {key: val.to(self.device) for key, val in data['new_sentence'].items()}

                data['video'] = data['video'].to(self.device)
                
                # text_embed, vid_embed, vid_embed_pooled = self.model(data, return_all_frames=True)

                result = self.model(data, return_all_frames=True)
                text_embed = result['text_embeds']
                noun_features = result['noun_features']
                vid_embed = result['video_features']
                vid_embed_pooled = result['video_features_pooled']
                # video_features_pooled_noun = result['video_features_pooled_noun']
                phrase_feat = result['phrase_feat']
                video_features_clip = result['video_features_clip']
                logits =result['logits']

                video_mean = result['video_mean']              


                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())

                """add"""
                phrase_embed_arr.append(phrase_feat.cpu())
                farm_embed_arr.append(video_features_clip.cpu())                

                """noun"""
                noun_embed_arr.append(noun_features.cpu())
                video_mean__embed_arr.append(video_mean.cpu())



                sims_batch1 = sim_matrix_training(text_embed, vid_embed_pooled, self.pooling_type)
                # sims_batch2 = sim_matrix_training(noun_features, video_features_pooled_noun, self.pooling_type)

                """add"""
                sims_batch = (logits + sims_batch1 ) * 0.5

                curr_loss = self.loss(sims_batch, self.model.clip.logit_scale)
                # curr_loss2 = self.loss(logits, self.model.clip.logit_scale)
                # curr_loss = (curr_loss1 + 0.5 * curr_loss2)


                total_val_loss += curr_loss.item()

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)
                
            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)
            """noun"""
            noun_embeds = torch.cat(noun_embed_arr)
            video_mean__embeds = torch.cat(video_mean__embed_arr)


            """add"""
            phrase_embed = torch.cat(phrase_embed_arr)
            farm_embed = torch.cat(farm_embed_arr)

            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]
            
            """add"""
            vid_embeds_per_video_id_farm = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id_farm:
                    vid_embeds_per_video_id_farm[v_id] = farm_embed[idx]            

            """video_mean"""
            vid_embeds_per_video_id_mean = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id_mean:
                    vid_embeds_per_video_id_mean[v_id] = video_mean__embeds[idx]  



            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

            """add"""
            farm_embeds = torch.stack([vid_embeds_per_video_id_farm[v_id] for v_id in vid_embeds_per_video_id_farm])

            """video_mean"""
            video_mean__embeds = torch.stack([vid_embeds_per_video_id_mean[v_id] for v_id in vid_embeds_per_video_id_mean])

             
            # Pool frames for inference once we have all texts and videos
            self.model.pool_frames.cpu()
            vid_embeds_pooled = self.model.pool_frames(text_embeds, vid_embeds)
            """noun"""
            # vid_embeds_pooled_noun = self.model.pool_frames(noun_embeds, vid_embeds)
            noun_embeds = F.normalize(noun_embeds, p=2, dim=-1)
            video_mean__embeds = F.normalize(video_mean__embeds, p=2, dim=-1)
            word_video_mean__logits = self.model.Logit_text_video(noun_embeds,video_mean__embeds)             

            # """"ddd"""              #不知道什么
            # vid_embeds_pooled_text = []
            # for idx, v_id in enumerate(all_vid_ids):
            #     text_embeds1 = text_embeds[idx].unsqueeze(0)
            #     vid_embeds_pooled_text1 = self.model.pool_frames(text_embeds1, vid_embeds)
            #     vid_embeds_pooled_text.append(vid_embeds_pooled_text1)
            # vid_embeds_pooled = torch.cat(vid_embeds_pooled_text,dim=1)

            """短语-帧"""
            phrase_embed = F.normalize(phrase_embed, p=2, dim=-1)
            farm_embeds = F.normalize(farm_embeds, p=2, dim=-1)
            vid_embeds = F.normalize(vid_embeds, p=2, dim=-1)
            word_frames_logits = self.model._attenion_over_fine_grained_sim_matrix(phrase_embed,vid_embeds) 
            logits_all = (word_frames_logits + word_video_mean__logits) * 0.5   
            # logits_all =   word_frames_logits

            # vid_embeds_pooled = self.model.pool_frames(text_embeds, vid_embeds)

            self.model.pool_frames.cuda()

            text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id(text_embeds, 
                    vid_embeds_pooled, all_vid_ids, self.pooling_type)

            
            sims1 = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)

            """add"""
            device = torch.device('cpu')
            logits_all = logits_all.to(device)
            a , b = logits_all.shape
            logits_all = logits_all.view(a,1,b)
            self.model._attenion_over_fine_grained_sim_matrix.cuda()
            sims = (sims1 + logits_all ) * 0.5


            total_val_loss = total_val_loss / len(self.valid_data_loader)

            metrics = self.metrics
            res = metrics(sims)
            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res
