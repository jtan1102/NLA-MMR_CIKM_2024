import os
from torch.utils.data import Dataset
import dill 
import pandas as pd
import dill
import numpy as np
import torch
import pickle
# the number of medications 112
# split mimic-iii dataset 
# split_point=9328
# eval_len=2072

def padding_visit_med(input_med_his, max_visit_num, save_path, pt_mode):
        """
        visit_med_embedding: (visit_num * emb)
        """
        #get the med_his embeddings
        text_dim = 768

        input_med_his = torch.LongTensor(input_med_his)
        med_his = []
        if max_visit_num == 2:
            if input_med_his.eq(-1).sum() != 0:
                med_his[0:2] = input_med_his[0:2]
            elif input_med_his.eq(-1).sum() == 0:
                med_his[0:2] = input_med_his[1:3]
        elif max_visit_num == 1:
            if input_med_his.eq(-1).sum() == 3 or input_med_his.eq(-1).sum() ==2:
                med_his[0:1] = input_med_his[0:1]
            elif input_med_his.eq(-1).sum() == 1:
                med_his[0:1] = input_med_his[1:2]
            elif input_med_his.eq(-1).sum() == 0:
                med_his[0:1] = input_med_his[2:3]
        else:
            med_his = input_med_his
        
        med_his = torch.LongTensor(med_his)
        mask= med_his.eq(-1)
        med_mask = mask.float().masked_fill(mask == 1, -1e9).masked_fill(mask == 0, float(0.0))
        med_embedding = torch.zeros((max_visit_num, text_dim)).float()

        for idx in range(max_visit_num):
            if med_his[idx] != -1:
                #records_med
                file_index = med_his[idx]
                save_file = os.path.join(save_path, f'{file_index}.pt')
                data_pt = torch.load(save_file)
                records_med_txt = data_pt[3].detach()
                med_embedding[idx] = records_med_txt

        return med_embedding, med_mask

class MIMIC_III_Datasets_text_final(Dataset):
    def __init__(self, root, pt_mode, max_visit_num=3):
        self.root = root
        self.med_his_file = dill.load(open("../data/MIMIC-III_data/raw/records_med_his.pkl",'rb'))
        self.save_path  = os.path.join(self.root, pt_mode)
        self.records_med_index = dill.load(open("../data/MIMIC-III_data/raw/records_med_index.pkl",'rb'))
        self.shift = 0
        split_point=9328

        self.med_index_list =  self.records_med_index[:split_point]
        self.max_visit_num = max_visit_num
        self.pt_mode = pt_mode

        return

    def __getitem__(self, index):

        save_file = os.path.join(self.save_path, f'{self.shift+index}.pt')
        data_pt = torch.load(save_file)

        records_dia_txt = data_pt[0].detach()
        records_pro_txt = data_pt[1].detach()
        records_sym_txt = data_pt[2].detach()
        records_med_txt = data_pt[3].detach()
        records_dps_txt = data_pt[4].detach()

        med_index = " ".join([str(elem) for elem in self.med_index_list[index]])

        m_list = self.med_index_list[index] 

        cur_med = torch.zeros(112)
        cur_med[m_list] = 1

        cur_med_ml = torch.full((112,), -1)
        cur_med_ml[:len(m_list)] = torch.LongTensor(m_list)

        his_med_embeddings, his_mask = padding_visit_med(self.med_his_file[self.shift+index], self.max_visit_num ,self.save_path, self.pt_mode)

        return records_dps_txt, records_med_txt, med_index, records_dia_txt, records_pro_txt, records_sym_txt, cur_med, cur_med_ml, his_med_embeddings, his_mask
    
    def __len__(self):
        return len(self.med_index_list)


class MIMIC_III_Datasets_text_Eval_final(Dataset):
    def __init__(self, root, pt_mode, max_visit_num=3):
        self.root = root
        self.med_his_file = dill.load(open("../data/MIMIC-III_data/raw/records_med_his.pkl",'rb'))
        self.records_med_index = dill.load(open("../data/MIMIC-III_data/raw/records_med_index.pkl",'rb'))
        self.max_visit_num = max_visit_num
        self.save_path  = os.path.join(self.root, pt_mode)
        self.pt_mode = pt_mode

        split_point=9328
        eval_len=2072
        self.med_index_list =  self.records_med_index[split_point+eval_len:]
        self.shift = split_point+eval_len

        return

    def __getitem__(self, index):
        file_index = index + self.shift
        save_file = os.path.join(self.save_path, f'{file_index}.pt')
        data_pt = torch.load(save_file)

        records_dia_txt = data_pt[0].detach()
        records_pro_txt = data_pt[1].detach()
        records_sym_txt = data_pt[2].detach()
        records_med_txt = data_pt[3].detach()
        records_dps_txt = data_pt[4].detach()

        med_index = " ".join([str(elem) for elem in self.med_index_list[index]])

        m_list = self.med_index_list[index] 
        cur_med = torch.zeros(112)
        cur_med[m_list] = 1
        cur_med_ml = torch.full((112,), -1)
        cur_med_ml[:len(m_list)] = torch.LongTensor(m_list)

        his_med_embeddings, his_mask = padding_visit_med(self.med_his_file[self.shift+index], self.max_visit_num, self.save_path, self.pt_mode)
            
        return records_dps_txt, records_med_txt, med_index, records_dia_txt, records_pro_txt, records_sym_txt, cur_med, cur_med_ml, his_med_embeddings, his_mask
    
    def __len__(self):
        return len(self.med_index_list)

class MIMIC_III_Datasets_text_Test_final(Dataset):
    def __init__(self, root, pt_mode, max_visit_num=3):
        self.root = root
        self.med_his_file = dill.load(open("../data/MIMIC-III_data/raw/records_med_his.pkl",'rb'))
        self.records_med_index = dill.load(open("../data/MIMIC-III_data/raw/records_med_index.pkl",'rb'))
        self.save_path  = os.path.join(self.root,pt_mode)
        self.max_visit_num = max_visit_num
        self.pt_mode = pt_mode

        split_point=9328
        eval_len=2072
        self.med_index_list =  self.records_med_index[split_point:split_point + eval_len]
        self.shift = split_point
        return

    def __getitem__(self, index):
        file_index = index + self.shift
        save_file = os.path.join(self.save_path, f'{file_index}.pt')
        data_pt = torch.load(save_file)

        records_dia_txt = data_pt[0].detach()
        records_pro_txt = data_pt[1].detach()
        records_sym_txt = data_pt[2].detach()
        records_med_txt = data_pt[3].detach()
        records_dps_txt = data_pt[4].detach()

        med_index = " ".join([str(elem) for elem in self.med_index_list[index]])

        m_list = self.med_index_list[index] 

        cur_med = torch.zeros(112)
        cur_med[m_list] = 1

        cur_med_ml = torch.full((112,), -1)
        cur_med_ml[:len(m_list)] = torch.LongTensor(m_list)

        his_med_embeddings, his_mask = padding_visit_med(self.med_his_file[self.shift+index], self.max_visit_num, self.save_path, self.pt_mode)
    
        return records_dps_txt, records_med_txt, med_index, records_dia_txt, records_pro_txt, records_sym_txt, cur_med, cur_med_ml, his_med_embeddings, his_mask
    
    def __len__(self):
        return len(self.med_index_list)
    

class MIMIC_III_Datasets_text_Test_final_my(Dataset):
    def __init__(self, index_list, root, pt_mode, max_visit_num=3):
        self.root = root
        self.med_his_file = dill.load(open("../data/MIMIC-III_data/raw/records_med_his.pkl",'rb'))
        self.records_med_index = dill.load(open("../data/MIMIC-III_data/raw/records_med_index.pkl",'rb'))
        self.index_list = index_list
        self.save_path  = os.path.join(self.root, pt_mode)
        self.max_visit_num = max_visit_num
        self.pt_mode = pt_mode

        split_point=9328
        eval_len=2072
        self.med_index_list =  self.records_med_index[split_point:split_point + eval_len]
        self.shift = split_point
        return

    def __getitem__(self, index):
        file_index = self.index_list[index] + self.shift

        save_file = os.path.join(self.save_path, f'{file_index}.pt')
        data_pt = torch.load(save_file)

        records_dia_txt = data_pt[0].detach()
        records_pro_txt = data_pt[1].detach()
        records_sym_txt = data_pt[2].detach()
        records_med_txt = data_pt[3].detach()
        records_dps_txt = data_pt[4].detach()

        med_index = " ".join([str(elem) for elem in self.med_index_list[self.index_list[index]]])

        m_list = self.med_index_list[self.index_list[index]] 

        cur_med = torch.zeros(112)
        cur_med[m_list] = 1

        cur_med_ml = torch.full((112,), -1)
        cur_med_ml[:len(m_list)] = torch.LongTensor(m_list)

        his_med_embeddings, his_mask = padding_visit_med(self.med_his_file[self.shift+self.index_list[index]], self.max_visit_num, self.save_path, self.pt_mode)

        return records_dps_txt, records_med_txt, med_index, records_dia_txt, records_pro_txt, records_sym_txt, cur_med, cur_med_ml, his_med_embeddings, his_mask
    
    def __len__(self):
        return len(self.index_list)
    

    


