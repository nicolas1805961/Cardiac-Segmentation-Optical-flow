import csv
import torch
from math import isinf, isnan

class GetStats(object):
    def __init__(self):
        self.ratio_list_attn = torch.zeros(size=(4,))
        self.ratio_list_mlp = torch.zeros(size=(4,))
        self.divider_attn = torch.full(size=(4,), fill_value=1e-7)
        self.divider_mlp = torch.full(size=(4,), fill_value=1e-7)
    
    def get_stats_attn(self, ratio, input_resolution):
        if input_resolution[0] == 56:
            if not isinf(ratio) and not isnan(ratio):
                self.ratio_list_attn[0] += ratio
                self.divider_attn[0] += 1.
        if input_resolution[0] == 28:
            if not isinf(ratio) and not isnan(ratio):
                self.ratio_list_attn[1] += ratio
                self.divider_attn[1] += 1.
        if input_resolution[0] == 14:
            if not isinf(ratio) and not isnan(ratio):
                self.ratio_list_attn[2] += ratio
                self.divider_attn[2] += 1.
        if input_resolution[0] == 7:
            if not isinf(ratio) and not isnan(ratio):
                self.ratio_list_attn[3] += ratio
                self.divider_attn[3] += 1.
    
    def get_stats_mlp(self, ratio, input_resolution):
        if input_resolution[0] == 56:
            if not isinf(ratio) and not isnan(ratio):
                self.ratio_list_mlp[0] += ratio 
                self.divider_mlp[0] += 1.
        if input_resolution[0] == 28:
            if not isinf(ratio) and not isnan(ratio):
                self.ratio_list_mlp[1] += ratio
                self.divider_mlp[1] += 1.
        if input_resolution[0] == 14:
            if not isinf(ratio) and not isnan(ratio):
                self.ratio_list_mlp[2] += ratio
                self.divider_mlp[2] += 1.
        if input_resolution[0] == 7:
            if not isinf(ratio) and not isnan(ratio):
                self.ratio_list_mlp[3] += ratio
                self.divider_mlp[3] += 1.
    
    def write_to_file(self):
        out_attn = (self.ratio_list_attn / self.divider_attn).tolist()
        out_mlp = (self.ratio_list_mlp / self.divider_mlp).tolist()
        with open('norm_stats.csv', 'w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([out_attn, out_mlp])

def init_globals():
    global global_iter
    global get_stats_object
    global_iter = 0
    get_stats_object = GetStats()