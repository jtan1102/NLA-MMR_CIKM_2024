import sys, os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '../'))
sys.path.append(os.path.join(current_directory, './gnn'))
sys.path.append(os.path.join(current_directory, '../datasets'))

import os
import time
import numpy as np
from tqdm import tqdm
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader
import pandas as pd
from torch_geometric.loader import DataLoader as pyg_DataLoader
import dill 
from copy import deepcopy
import os
from common_layers import *
from gnn import GNNGraph
from gnn import graph_batch_from_smile
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params
import time
from collections import defaultdict
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
import math
import wandb

from datasets import (
    MIMIC_III_Datasets_text_final,  MIMIC_III_Datasets_text_Eval_final, MIMIC_III_Datasets_text_Test_final,MIMIC_III_Datasets_text_Test_final_my,
    MIMIC_IV_Datasets_text_final,  MIMIC_IV_Datasets_text_Eval_final, MIMIC_IV_Datasets_text_Test_final, MIMIC_IV_Datasets_text_Test_final_my,
)

def save_model(save_best, epoch=None, model_name=""):
    if args.output_model_dir is not None:
        if model_name != "":
            model_file =  "model_{}.pth".format(model_name)

        epoch = str(epoch)
        if not os.path.exists(os.path.join(args.output_model_dir, epoch)):
            os.makedirs(os.path.join(os.path.join(args.output_model_dir, epoch)))

        text2latent_saved_file_path = os.path.join(args.output_model_dir, epoch,"text2latent_{}".format(model_file))
        torch.save(text2latent.state_dict(), text2latent_saved_file_path)

        mol2latent_saved_file_path = os.path.join(args.output_model_dir, epoch, "mol2latent_{}".format(model_file))
        torch.save(mol2latent.state_dict(), mol2latent_saved_file_path)

        aggregator_saved_file_path = os.path.join(args.output_model_dir, epoch, "aggregator_{}".format(model_file))
        torch.save(aggregator.state_dict(), aggregator_saved_file_path)

        dsp_aggregator_saved_file_path = os.path.join(args.output_model_dir, epoch, "dsp_aggregator_{}".format(model_file))
        torch.save(dsp_aggregator.state_dict(), dsp_aggregator_saved_file_path)

        proj_sdp_saved_file_path = os.path.join(args.output_model_dir, epoch, "proj_dsp_{}".format(model_file))
        torch.save(proj_dsp.state_dict(), proj_sdp_saved_file_path)

        global_encoder_saved_file_path = os.path.join(args.output_model_dir, epoch, "global_encoder_{}".format(model_file))
        torch.save(global_encoder.state_dict(), global_encoder_saved_file_path)

    return

@torch.no_grad()
def eval_epoch(dataloader, batch_size, med_vocab_size, input_med_rep):
    criterion = binary_cross_entropy_with_logits
    text2latent.eval()
    mol2latent.eval()
    aggregator.eval()
    global_encoder.eval()

    dsp_aggregator.eval()
    proj_dsp.eval()

    accum_loss_multi, accum_loss_bce, accum_loss_rec= 0, 0,0
    smm_record, ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    visit_cnt = 0
    med_cnt = 0

    print("Start testing!")
    start_time = time.time()
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    smm_record = []
    for step, batch in enumerate(L):
        y_gt, y_pred, y_pred_prob, y_pred_label = [[] for _ in range(4)]
        batch_num = batch[0].size(0)
        
        global_embeddings = global_encoder(**MPNN_drug_data)
        global_embeddings = torch.mm(average_projection, global_embeddings)

        records_dps_txt = batch[0].to(device)
        records_med_txt = batch[1].to(device)
        med_index = batch[2]
        records_dia_txt = batch[3].to(device)
        records_pro_txt = batch[4].to(device)
        records_sym_txt = batch[5].to(device)
        bce_target =  batch[6].to(device)
        multi_target = batch[7].to(device)
        
        his_med_embeddings = batch[8].to(device)
        his_mask = batch[9].to(device)

        med_index = [list(map(int,  elem.split(" "))) for elem in med_index]
        
        description_repr = records_dps_txt 
        
        description_repr_input = description_repr

        attn_dsp = dsp_aggregator(description_repr_input,torch.cat([torch.unsqueeze(records_sym_txt,1),torch.unsqueeze(records_dia_txt,1),torch.unsqueeze(records_pro_txt,1)],dim=1),mask=None)

        description_repr = torch.cat([description_repr_input, attn_dsp],dim=-1)

        description_repr =  proj_dsp(description_repr)

        his_med_repr,_ = aggregator(
        description_repr, his_med_embeddings, mask=his_mask)

        description_repr = text2latent(description_repr)
        
        # GNN+LLM => encode medications
        med_rep = torch.cat([input_med_rep, global_embeddings],dim=1)

        med_rep = mol2latent(med_rep)  

        description_repr = description_repr + his_med_repr
        output = torch.mm(description_repr, med_rep.t()) 
        
        y_gt_tmp = torch.zeros((batch_num, med_vocab_size)).numpy()

        for index in range(batch_num):
            y_gt_tmp[index, med_index[index]] = 1
     
        for i in range(len(y_gt_tmp)):
            y_gt.append(y_gt_tmp[i])

        loss_bce = binary_cross_entropy_with_logits(output, bce_target)
        loss_multi = multilabel_margin_loss(torch.sigmoid(output), multi_target)


        output = torch.sigmoid(output).detach().cpu().numpy()
        for i in range(len(output)):
            y_pred_prob.append(output[i])

        accum_loss_multi += loss_multi.item()
        accum_loss_bce += loss_bce.item()

        loss_rec = args.bce_weight * loss_bce + (1-args.bce_weight) * loss_multi
        accum_loss_rec += loss_rec.item()

        y_pred_tmp_mat = copy.deepcopy(output)
        
        for i in range(len(y_pred_tmp_mat)):
            y_pred_tmp = y_pred_tmp_mat[i]

            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label_tmp = sorted(y_pred_label_tmp)
            y_pred_label.append(y_pred_label_tmp)
            smm_record.append([y_pred_label_tmp])
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        # llprint('\rtest step: {} / {}'.format(step + 1, len(L)))

    accum_loss_multi /= len(L)
    accum_loss_bce /= len(L)
    accum_loss_rec /= len(L)

    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)

    output_str = '\nDDI Rate: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, ' +\
        'AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'
    llprint(output_str.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    print("REC Loss: {:.5f}\tBCE loss:{:.5f}\t Multi loss:{:.5f}\tTime: {:.5f}".format(accum_loss_rec, accum_loss_bce, accum_loss_multi, time.time() - start_time))

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt, accum_loss_rec, accum_loss_bce, accum_loss_multi


def train(
    epoch,
    dataloader,batch_size,  bce_weight, med_vocab_size, input_med_rep):

    criterion = binary_cross_entropy_with_logits

    text2latent.train()
    mol2latent.train()
    aggregator.train()
    global_encoder.train()

    dsp_aggregator.train()
    proj_dsp.train()

    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    print("Start training!")
    start_time = time.time()
    accum_loss_rec= 0
    accum_loss_multi, accum_loss_bce= 0, 0

    for step, batch in enumerate(L):
  
        global_embeddings = global_encoder(**MPNN_drug_data)
        global_embeddings = torch.mm(average_projection, global_embeddings)
        
        records_dps_txt = batch[0].to(device)
        records_med_txt = batch[1].to(device)
        med_index = batch[2]
        records_dia_txt = batch[3].to(device)
        records_pro_txt = batch[4].to(device)
        records_sym_txt = batch[5].to(device)

        bce_target =  batch[6].to(device)
        multi_target = batch[7].to(device)

        his_med_embeddings = batch[8].to(device)
        his_mask = batch[9].to(device)

        med_index = [list(map(int, elem.split(" "))) for elem in med_index]
        
        description_repr_input = records_dps_txt 
        
        attn_dsp = dsp_aggregator(description_repr_input,torch.cat([torch.unsqueeze(records_sym_txt,1),torch.unsqueeze(records_dia_txt,1),torch.unsqueeze(records_pro_txt,1)],dim=1),mask=None)

        description_repr = torch.cat([description_repr_input, attn_dsp],dim=-1)

        description_repr =  proj_dsp(description_repr)
            
        his_med_repr,_= aggregator(
        description_repr, his_med_embeddings, mask=his_mask)
        description_repr = text2latent(description_repr)

        # GNN+LLM => encode medications
        med_rep = torch.cat([input_med_rep, global_embeddings],dim=1)
          
        med_rep = mol2latent(med_rep)  

        description_repr = description_repr + his_med_repr

        result = torch.mm(description_repr, med_rep.t())  

        sigmoid_res = torch.sigmoid(result)

        loss_bce = binary_cross_entropy_with_logits(result, bce_target)
        loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)

        loss_rec = bce_weight * loss_bce + (1-bce_weight) * loss_multi
        accum_loss_rec += loss_rec.item()

        accum_loss_multi += loss_multi.item()
        accum_loss_bce += loss_bce.item()

        optimizer.zero_grad()
        loss_rec.backward()
        optimizer.step()

        if step % 1000 == 0:
            print('\rtraining step: {} / {}, REC loss: {:.4f},   loss_bce: {:.4f}, loss_multi: {:.4f} '
                    .format(step, len(L), loss_rec,  loss_bce, loss_multi))

    accum_loss_rec /= len(L)
    accum_loss_multi /= len(L)
    accum_loss_bce /= len(L)
    
    print("REC Loss: {:.5f}\tBCE Loss:{:.5f}\tMulti Loss:{:.5f}\tTime: {:.5f}".format(accum_loss_rec, accum_loss_bce, accum_loss_multi, time.time() - start_time))

    return  accum_loss_rec, accum_loss_multi, accum_loss_bce

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--dataspace_path", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="MIMIC-III")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_scale", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument("--output_model_dir", type=str, default=None)

    #for GNN 
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--dropout_ratio", type=float, default=0.1)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')
    parser.add_argument("--gnn_dim", type=int, default=64)
    parser.add_argument("--gnn_dp", type=float, default=0.7)

    parser.add_argument("--K_emb_dim", type=int, default=256)
    parser.add_argument('--data_file_name', type=str, default='../data/MIMIC-III_data/raw/records_text_iii.pkl')
    
    parser.add_argument("--bce_weight", type=float, default=0.95)
    parser.add_argument("--med_vocab_size", type=int, default=112)

    # max_visit_num: the # of past records in historical information modeling
    parser.add_argument("--max_visit_num", type=int, default=3)  
    parser.add_argument("--pt_mode", type=str, default="bio_pt", choices=["sci_pt", "clinical_pt","openai_pt","pub_pt","bio_pt","bleu_pt"])

    parser.add_argument("--patient_sample",type=str, default="True")
    parser.add_argument("--Train", action="store_true")
    
    text_dim = 768
    args = parser.parse_args()
    if args.dataset == "MIMIC-III":
        args.med_vocab_size =112
    elif args.dataset == "MIMIC-IV":
        args.med_vocab_size =121
    print("arguments\t", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if "MIMIC-III" in args.dataset:
        dataset_root = os.path.join(args.dataspace_path, "MIMIC-III_data")
    elif "MIMIC-IV" in args.dataset:
        dataset_root = os.path.join(args.dataspace_path, "MIMIC-IV_data")
    else:
        assert False and "invlaid dataset name"
    kwargs = {}

    if "MIMIC-III" in args.dataset:
        molecule_path = os.path.join(dataset_root , "raw", "atc3toSMILES_iii.pkl")
        voc_path = os.path.join(dataset_root , "raw", "voc_iii_sym1_mulvisit.pkl")
        ddi_adj_path = os.path.join(dataset_root , "raw", "ddi_A_iii.pkl")
        ori_data_path = os.path.join(dataset_root , "raw", "records_ori_iii.pkl")
    elif "MIMIC-IV" in args.dataset:
        molecule_path = os.path.join(dataset_root , "raw", "atc3toSMILES_iv.pkl")
        voc_path = os.path.join(dataset_root , "raw", "voc_iv_sym1_mulvisit.pkl")
        ddi_adj_path = os.path.join(dataset_root , "raw", "ddi_A_iv.pkl")
        ori_data_path = os.path.join(dataset_root , "raw", "records_ori_iv.pkl")

    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)

    with open(ori_data_path, 'rb') as Fin:
        origin_data = dill.load(Fin)

    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)

    diag_voc, pro_voc, med_voc = \
        voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )

    average_projection, smiles_list = \
    buildPrjSmiles(molecule, med_voc.idx2word)
    average_projection = average_projection.to(device)
    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {'batched_data': molecule_graphs.to(device)}
    molecule_para = {
        'num_layer': args.num_layer, 'emb_dim': args.gnn_dim, 'graph_pooling': args.graph_pooling,
        'drop_ratio': args.gnn_dp, 'gnn_type': 'gin', 'virtual_node': False
    }

    MPNN_drug_data = molecule_forward

    if args.dataset == "MIMIC-III":
        train_dataset =MIMIC_III_Datasets_text_final(root = dataset_root, pt_mode = args.pt_mode,  max_visit_num=args.max_visit_num)
        valid_dataset = MIMIC_III_Datasets_text_Eval_final(root = dataset_root, pt_mode = args.pt_mode,  max_visit_num=args.max_visit_num)
        test_dataset = MIMIC_III_Datasets_text_Test_final(root = dataset_root, pt_mode = args.pt_mode,  max_visit_num=args.max_visit_num)
    elif args.dataset == "MIMIC-IV":
        train_dataset =MIMIC_IV_Datasets_text_final(root = dataset_root, pt_mode = args.pt_mode,  max_visit_num=args.max_visit_num)
        valid_dataset = MIMIC_IV_Datasets_text_Eval_final(root = dataset_root, pt_mode = args.pt_mode,  max_visit_num=args.max_visit_num)
        test_dataset = MIMIC_IV_Datasets_text_Test_final(root = dataset_root, pt_mode = args.pt_mode,  max_visit_num=args.max_visit_num)

    dataloader_class = torch_DataLoader

    train_loader = dataloader_class(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last = True)
    val_loader = dataloader_class(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last = False)
    test_loader = dataloader_class(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last = False)
    
    global_encoder = GNNGraph(**molecule_para).to(device)
    
    dsp_aggregator = AdjAttenAgger_no_projection(
            text_dim, text_dim, args.K_emb_dim, device
        ).to(device)
    
    proj_dsp = nn.Sequential(
            nn.Linear(2*text_dim, text_dim),
            nn.GELU(),
            nn.Dropout(args.dropout_ratio),
        ).to(device)
    
    aggregator = AdjAttenAgger(
            text_dim, text_dim, args.K_emb_dim, device
        ).to(device)

    text2latent = MLP(text_dim, [args.K_emb_dim, args.K_emb_dim],batch_norm=False, activation="gelu", dropout=args.dropout_ratio).to(device)
    mol2latent = MLP(text_dim+64, [args.K_emb_dim, args.K_emb_dim],batch_norm=False, activation="gelu", dropout=args.dropout_ratio).to(device)
   
    model_param_group = [
            {"params": text2latent.parameters(), "lr": args.lr * args.lr_scale},
            {"params": mol2latent.parameters(), "lr": args.lr * args.lr_scale},
            {"params": aggregator.parameters(), "lr": args.lr * args.lr_scale},
            {"params": global_encoder.parameters(), "lr": args.lr * args.lr_scale},
            {"params": dsp_aggregator.parameters(), "lr": args.lr * args.lr_scale},
            {"params": proj_dsp.parameters(), "lr": args.lr * args.lr_scale},
        ]
        



    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
    
    print("parameters", get_n_params(text2latent)+get_n_params(mol2latent)+get_n_params(aggregator)+get_n_params(global_encoder)+
          get_n_params(dsp_aggregator)+get_n_params(proj_dsp))

    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    best_epoch = 0
    best_ja=0

    kwargs['batch_size']= args.batch_size
    kwargs['bce_weight']= args.bce_weight
    kwargs['med_vocab_size'] = args.med_vocab_size

    save_file = os.path.join(dataset_root,args.pt_mode,"med_output_tensor.pt")

    input_med_rep = torch.load(save_file)
    input_med_rep = input_med_rep.detach().to(device)

    kwargs['input_med_rep'] = input_med_rep

    if args.Train:
        best_ja = 0
        for e in range(1, args.epochs+1):
            print("---------Epoch {}-----------".format(e))
    
            accum_loss_rec, accum_loss_multi, accum_loss_bce = train(e, train_loader,  **kwargs)
            
            model_name = 'Epoch_{}'.format(e)
            print("Valid perfomance")
            ddi_rate_eval, ja_eval, prauc_eval, avg_p_eval, avg_r_eval, avg_f1_eval, avg_med_eval, accum_loss_rec, accum_loss_bce, accum_loss_multi = eval_epoch(val_loader, args.batch_size, args.med_vocab_size, input_med_rep)

            if best_ja < ja_eval:
                best_epoch = e
                best_ja = ja_eval
                save_model(save_best=True, epoch=e, model_name=model_name)
            print("best_ja",best_ja)
            print("best_epoch",best_epoch)

        save_best_epoch = best_epoch

        text2latent.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(save_best_epoch),"text2latent_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading text2latent")
        mol2latent.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(save_best_epoch), "mol2latent_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading mol2latent")
        aggregator.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(save_best_epoch), "aggregator_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading aggregator")
        dsp_aggregator.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(save_best_epoch), "dsp_aggregator_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading dsp_aggregator")
        proj_dsp.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(save_best_epoch), "proj_dsp_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading proj_dsp_aggregator")
        global_encoder.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(save_best_epoch), "global_encoder_model_Epoch_{}.pth".format(str(save_best_epoch))),'rb'), map_location=torch.device('cpu')))
        print("loading MPNN")

        result = []

        if args.patient_sample=="True" and (args.dataset == "MIMIC-III" or "MIMIC-IV"):
            patient_list = []
            initial_patient_index = 0
            split_point = int(len(origin_data) * 2 / 3) 
            eval_len = int(len(origin_data[split_point:]) / 2)
            origin_data_test = origin_data[split_point:split_point + eval_len]
            for patient in origin_data_test:
                every_patient = []
                for visit in patient:
                    every_patient.append(initial_patient_index)
                    initial_patient_index = initial_patient_index + 1
                patient_list.append(every_patient)

        for _ in range(10):
            if args.patient_sample=="True" and args.dataset == "MIMIC-III":
                sample_size = np.random.choice(a=len(origin_data_test), size= round(len(origin_data_test) * 0.8), replace=True)
                test_sample = list(sample_size)
                random_list = []
                for sample_idx in test_sample:
                    random_list= random_list + patient_list[sample_idx]
            
                test_dataset_my = MIMIC_III_Datasets_text_Test_final_my(index_list=random_list, root = dataset_root, pt_mode = args.pt_mode,  max_visit_num=args.max_visit_num)
                test_loader_my = dataloader_class(test_dataset_my, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            elif args.patient_sample=="True" and args.dataset == "MIMIC-IV":
                sample_size = np.random.choice(a=len(origin_data_test), size= round(len(origin_data_test) * 0.8), replace=True)
                test_sample = list(sample_size)
                random_list = []
                for sample_idx in test_sample:
                    random_list= random_list + patient_list[sample_idx]
                test_dataset_my = MIMIC_IV_Datasets_text_Test_final_my(index_list=random_list, root = dataset_root, pt_mode = args.pt_mode,  max_visit_num=args.max_visit_num)
                test_loader_my = dataloader_class(test_dataset_my, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            else:
                test_sampler_my = torch.utils.data.RandomSampler(data_source=test_dataset,replacement=True, num_samples=round(len(test_dataset) * 0.8))
                test_loader_my = dataloader_class(dataset=test_dataset,sampler=test_sampler_my, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
     
            with torch.set_grad_enabled(False):
                ddi_rate, ja, prauc, avg_p_eval, avg_r_eval, avg_f1, avg_med, accum_loss_rec, accum_loss_bce, accum_loss_multi = eval_epoch(test_loader_my, args.batch_size, args.med_vocab_size, input_med_rep)
                result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print (outstring)

