# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from .modules.brep_encoder import BrepEncoder
from .modules.seq_decoder import SeqDecoder
from .modules.utils.loss import CadLoss, DiffLoss, DomainDiscriminator, DomainAdversarialLoss
from .modules.utils.vec2json import convert2json
from .modules.utils.macro import EOS_IDX, MAX_PRIM, MAX_FEAT

class Brep2Seq(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the model.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.prim_encoder = BrepEncoder(
            n_indeg=64,  
            n_outdeg=64,  
            n_spatial=128,  
            n_edge_dist=128,  
            edge_type="multi_hop",  
            mhm_dist=8,  
            n_layer=args.n_encoder_layer,  
            d_embedding=args.d_z,  
            d_ff_embedding=args.d_feedforward,  
            n_head=args.n_head,  
            dropout=args.dropout,  
            attn_dropout=args.attn_dropout,  
            act_dropout=args.act_dropout,  
            act_fn="gelu",  
        )

        self.feat_encoder = BrepEncoder(
            n_indeg=64,  
            n_outdeg=64,  
            n_spatial=128,  
            n_edge_dist=128,  
            edge_type="multi_hop",  
            mhm_dist=8,  
            n_layer=args.n_encoder_layer,  
            d_embedding=args.d_z,  
            d_ff_embedding=args.d_feedforward,  
            n_head=args.n_head,  
            dropout=args.dropout,  
            attn_dropout=args.attn_dropout,  
            act_dropout=args.act_dropout,  
            act_fn="gelu",  
        )

        self.seq_decoder = SeqDecoder(
            n_head=args.n_head,
            n_layer=args.n_decoder_layer,
            d_model=args.d_model,
            d_z=args.d_z,
            d_ff=args.d_feedforward,
            dropout=args.dropout,
        )

        self.loss_weights = 2.0
        self.loss_prim = CadLoss(type="primitive")
        self.loss_feat = CadLoss(type="feature")
        self.loss_comm = DomainAdversarialLoss(
            discriminator=DomainDiscriminator(args.d_z, hidden_size=1024)
        )
        self.loss_diff1 = DiffLoss()
        self.loss_diff2 = DiffLoss()
        self.loss_diff3 = DiffLoss()

        self.acc_prim_cmd = []
        self.acc_prim_param = []
        self.acc_feat_cmd = []
        self.acc_feat_param = []

    def training_step(self, batch, batch_idx):
        self.prim_encoder.train()
        self.feat_encoder.train()
        self.seq_decoder.train()

        _, z_p = self.prim_encoder(batch)
        _, z_f = self.feat_encoder(batch)
        z = z_p + z_f
        output = self.seq_decoder(z)

        # Loss recons
        prim_loss = self.loss_prim(output, batch["label_prim_cmd"], batch["label_prim_param"], self.loss_weights)
        prim_loss = sum(prim_loss.values())
        self.log("loss_primitive", prim_loss, on_step=True, on_epoch=True)
        feat_loss = self.loss_feat(output, batch["label_feat_cmd"], batch["label_feat_param"], self.loss_weights)
        feat_loss = sum(feat_loss.values())
        self.log("loss_feature", feat_loss, on_step=True, on_epoch=True)

        # Loss similarity & differenec
        z_p_0, z_p_1 = z_p.chunk(2, dim=0)
        z_f_0, z_f_1 = z_f.chunk(2, dim=0)
        self.loss_comm.train()
        comm_loss = self.loss_comm(z_p_0, z_p_1)
        self.log("loss_similarity", comm_loss, on_step=True, on_epoch=True)
        diff_loss1 = self.loss_diff1(z_f_0, z_f_1)
        diff_loss2 = self.loss_diff2(z_p_0, z_f_0)
        diff_loss3 = self.loss_diff3(z_p_1, z_f_1)
        diff_loss = diff_loss1 + diff_loss2 + diff_loss3
        self.log("loss_difference", diff_loss, on_step=True, on_epoch=True)

        loss = prim_loss + feat_loss + 0.25 * comm_loss + 0.25 * diff_loss
        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.prim_encoder.eval()
        self.feat_encoder.eval()
        self.seq_decoder.eval()

        _, z_p = self.prim_encoder(batch)
        _, z_f = self.feat_encoder(batch)
        z = z_p + z_f
        output = self.seq_decoder(z)

        # Acc
        self.cal_acc_primitive(output, batch["label_prim_cmd"], batch["label_prim_param"])
        self.cal_acc_feature(output, batch["label_feat_cmd"], batch["label_feat_param"])

        # Loss recons
        prim_loss = self.loss_prim(output, batch["label_prim_cmd"], batch["label_prim_param"], self.loss_weights)
        prim_loss = sum(prim_loss.values())
        feat_loss = self.loss_feat(output, batch["label_feat_cmd"], batch["label_feat_param"], self.loss_weights)
        feat_loss = sum(feat_loss.values())
        loss = prim_loss + feat_loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        self.logger.experiment.add_scalars('acc_primitive',
                                           {"opr": np.mean(self.acc_prim_cmd),
                                            "param": np.mean(self.acc_prim_param)},
                                           global_step=self.current_epoch)

        self.logger.experiment.add_scalars('acc_feature',
                                           {"opr": np.mean(self.acc_feat_cmd),
                                            "param": np.mean(self.acc_feat_param)},
                                           global_step=self.current_epoch)
        self.acc_prim_cmd = []
        self.acc_prim_param = []
        self.acc_feat_cmd = []
        self.acc_feat_param = []

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.prim_encoder.eval()
        self.feat_encoder.eval()
        self.seq_decoder.eval()

        _, z_p = self.prim_encoder(batch)
        _, z_f = self.feat_encoder(batch)
        z = z_p + z_f
        output = self.seq_decoder(z)

        # output predicted result-------------------------------------------------------------------
        prim_cmd = torch.argmax(torch.softmax(output['prim_cmd'], dim=-1), dim=-1)  # (N, S)
        prim_param = torch.argmax(torch.softmax(output['prim_param'], dim=-1), dim=-1) - 1  # (N, S, n_args)
        prim_cmd = prim_cmd.long().detach().cpu().numpy()  # (N, S)
        prim_param = prim_param.long().detach().cpu().numpy()  # (N, S, n_args)

        feat_cmd = torch.argmax(torch.softmax(output['feat_cmd'], dim=-1), dim=-1)  # (N, S)
        feat_param = torch.argmax(torch.softmax(output['feat_param'], dim=-1), dim=-1) - 1  # (N, S, n_args)
        feat_cmd = feat_cmd.long().detach().cpu().numpy()  # (N, S)
        feat_param = feat_param.long().detach().cpu().numpy()  # (N, S, n_args)

        # 将结果转为json文件--------------------------------------------------------------------------
        batch_size = np.size(prim_cmd, 0)
        for i in range(batch_size):
            end_pos = MAX_PRIM - np.sum((prim_cmd[i][:] == EOS_IDX).astype(np.int))
            primitive_type = prim_cmd[i][:end_pos + 1]  # (Seq)
            primitive_param = prim_param[i][:end_pos + 1][:]  # (Seq, n_args)

            end_pos = MAX_FEAT - np.sum((feat_cmd[i][:] == EOS_IDX).astype(np.int))
            feature_type = feat_cmd[i][:end_pos + 1]  # (Seq)
            feature_param = feat_param[i][:end_pos + 1][:]  # (Seq, n_args)

            file_name = "rebuild_{}.json".format(str(i))
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/prediction")
            if not os.path.exists(file_path): os.makedirs(file_path)
            convert2json(primitive_type, primitive_param, feature_type, feature_param, os.path.join(file_path,file_name))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            threshold=0.0001, 
            threshold_mode='rel',
            min_lr=0.000001, 
            cooldown=2, 
            verbose=False
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch", 
                "frequency": 1, 
                "monitor": "val_loss"
            }
        }

    def optimizer_step(self,
                        epoch,
                        batch_idx,
                        optimizer,
                        optimizer_idx,
                        optimizer_closure,
                        on_tpu,
                        using_native_amp,
                        using_lbfgs,
                        ):
        # update params
        optimizer.step(closure=optimizer_closure)
        # manually warm up lr without a scheduler
        if self.trainer.global_step < 10000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 10000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 0.0001

    def cal_acc_primitive(self, output, label_commands, label_args):
        out_commands = torch.argmax(torch.softmax(output['prim_cmd'], dim=-1), dim=-1)
        out_commands = out_commands.long().detach().cpu().numpy()  # (N, S)
        out_args = torch.argmax(torch.softmax(output['prim_param'], dim=-1), dim=-1) - 1
        out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)
        gt_commands = label_commands.squeeze(1).long().detach().cpu().numpy()  # (N, S)
        gt_args = label_args.squeeze(1).long().detach().cpu().numpy()  # (N, S, n_args)

        # acc_opr
        commands_comp = (out_commands == gt_commands).astype(np.int)
        self.acc_prim_cmd.append(np.mean(commands_comp))

        # acc_param
        acc_pos = np.where(out_commands == gt_commands)
        args_comp = (np.abs(out_args - gt_args) < 5).astype(np.int)
        self.acc_prim_param.append(np.mean(args_comp[acc_pos]))

    def cal_acc_feature(self, output, label_commands, label_args):
        out_commands = torch.argmax(torch.softmax(output['feat_cmd'], dim=-1), dim=-1)
        out_commands = out_commands.long().detach().cpu().numpy()  # (N, S)
        out_args = torch.argmax(torch.softmax(output['feat_param'], dim=-1), dim=-1) - 1
        out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)
        gt_commands = label_commands.squeeze(1).long().detach().cpu().numpy()  # (N, S)
        gt_args = label_args.squeeze(1).long().detach().cpu().numpy()  # (N, S, n_args)

        # acc opr
        commands_comp = (out_commands == gt_commands).astype(np.int)
        self.acc_feat_cmd.append(np.mean(commands_comp))

        # acc param
        acc_pos = np.where(out_commands == gt_commands)
        args_comp = (np.abs(out_args - gt_args) < 5).astype(np.int)
        self.acc_feat_param.append(np.mean(args_comp[acc_pos]))
        
