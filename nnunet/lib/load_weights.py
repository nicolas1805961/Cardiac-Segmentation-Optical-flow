import torch
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

def load_weights(model, config, starting_layer_number):
    new_dict = load_weights_enc_dec(model, config['transformer_type'], config['transformer_depth'], config['device'], config['rpe_mode'], config['proj'], starting_layer_number=starting_layer_number)
    load_weights_bottleneck(model, new_dict, config['bottleneck'], config['transformer_depth'], config['num_bottleneck_layers'], config['device'], config['rpe_mode'], config['proj'], config['dim_feedforward'])
    return model

def load_weights_enc_dec(model, transformer_type, transformer_depth, device, rpe_mode, proj, starting_layer_number):
    if transformer_type == 'swin':
        return load_weights_enc_dec_swin(model, 'swin_tiny_patch4_window7_224.pth', transformer_depth, device, rpe_mode, starting_layer_number=starting_layer_number)
    elif transformer_type == 'vit':
        return load_weights_enc_dec_vit(model, 'deit_base_patch16_224_ctx_product_50_shared_qkv.pth', transformer_depth, device, rpe_mode, proj, starting_layer_number=starting_layer_number)

def load_weights_bottleneck(model, new_dict, bottleneck, transformer_depth, bottleneck_depth, device, rpe_mode, proj, dim_feedforward):
    if bottleneck == 'swin':
        load_swin_bottleneck(model, new_dict, 'swin_tiny_patch4_window7_224.pth', bottleneck_depth, device, rpe_mode, dimension='2d')
    elif bottleneck == 'swin_3d':
        load_swin_bottleneck(model, new_dict, 'swin_tiny_patch4_window7_224.pth', bottleneck_depth, device, rpe_mode, dimension='3d')
    elif bottleneck == 'vit':
        load_vit_bottleneck(model, new_dict, 'deit_base_patch16_224_ctx_product_50_shared_qkv.pth', transformer_depth, bottleneck_depth, device, rpe_mode, proj)
    elif bottleneck == 'vit_3d':
        load_vit_bottleneck(model, new_dict, 'deit_base_patch16_224_ctx_product_50_shared_qkv.pth', transformer_depth, bottleneck_depth, device, rpe_mode, proj)
    elif bottleneck == 'factorized':
        load_weights_ifc_bottleneck(model, new_dict, 'r101.pth', 'deit_base_patch16_224_ctx_product_50_shared_qkv.pth', rpe_mode, bottleneck_depth, transformer_depth, dim_feedforward, proj)
        

def replace_rpe_vit(x, rpe_mode):
    if rpe_mode == 'contextual':
        x = x.replace('rpe_q.lookup_table_weight', 'q_rpe_table')
        x = x.replace('rpe_k.lookup_table_weight', 'k_rpe_table')
        x = x.replace('rpe_v.lookup_table_weight', 'v_rpe_table')
    elif rpe_mode == 'bias':
        x = x.replace('rpe_q.lookup_table_weight', 'rpe_table')
        x = x.replace('rpe_k.lookup_table_weight', 'rpe_table')
        x = x.replace('rpe_v.lookup_table_weight', 'rpe_table')
    return x

def replace_rpe_swin_contextual(x):
    q_table = x.replace('relative_position_bias_table', 'q_rpe_table')
    k_table = x.replace('relative_position_bias_table', 'k_rpe_table')
    v_table = x.replace('relative_position_bias_table', 'v_rpe_table')
    return q_table, k_table, v_table

def replace_vit(new_k):
    new_k = new_k.replace('attn', 'self_attn')
    new_k = new_k.replace('mlp.fc1', 'linear1')
    new_k = new_k.replace('mlp.fc2', 'linear2')
    new_k = new_k.replace('qkv.weight', 'in_proj_weight')
    new_k = new_k.replace('qkv.bias', 'in_proj_bias')
    new_k = new_k.replace('proj.weight', 'out_proj.weight')
    new_k = new_k.replace('proj.bias', 'out_proj.bias')
    return new_k

def replace_vit_encoder(k, model_starting_layer, vit_block_number, model_nb_blocks_per_layer, rpe_mode):
    model_layer_number = model_starting_layer + vit_block_number//model_nb_blocks_per_layer
    model_block_number = vit_block_number % model_nb_blocks_per_layer
    new_k = '.'.join(['encoder', 'layers', str(model_layer_number), 'blocks', str(model_block_number)] + k.split('.')[2:])
    new_k = replace_vit(new_k)
    new_k = replace_rpe_vit(new_k, rpe_mode)
    return new_k

def replace_vit_bottleneck(k, block_number, bottleneck_depth, rpe_mode):
    model_block_number = block_number % bottleneck_depth
    new_k = '.'.join(['bottleneck', 'blocks', str(model_block_number)] + k.split('.')[2:])
    new_k = replace_vit(new_k)
    new_k = replace_rpe_vit(new_k, rpe_mode)
    return new_k

def replace_swin_encoder(k, swin_layer_nb, swin_block_number, starting_layer_nb):
    if starting_layer_nb == 0:
        model_layer_number = swin_layer_nb
    elif starting_layer_nb == 2:
        model_layer_number = swin_layer_nb + 2
    new_k = '.'.join(['encoder', 'layers', str(model_layer_number), 'blocks', str(swin_block_number)] + k.split('.')[4:])
    return new_k

def replace_vit_swin_decoder(k, starting_layer_number):
    if starting_layer_number == 0:
        model_layer_number = abs(int(k.split('.')[2]) - 2)
    elif starting_layer_number == 2:
        model_layer_number = abs(int(k.split('.')[2]) - 4)
    model_block_number = int(k.split('.')[4])
    new_k = '.'.join(['decoder', 'layers', str(model_layer_number), 'blocks', str(model_block_number)] + k.split('.')[5:])
    return new_k

def check_conditions_vit(k, proj, rpe_mode):
    if proj == 'conv' and ('qkv.weight' in k or 'qkv.bias' in k):
        return True
    if rpe_mode not in ['bias', 'contextual'] and 'lookup_table_weight' in k:
        return True
    return False

def check_conditions_unexpected_swin(k):
    to_check = ['.relative_position_index', '.qkv.']
    if all(x not in k for x in to_check):
        return True

def check_conditions_missing_swin(k):
    to_check = ['.bn', '.conv', '.downsample', '.concat_back_dim', '.upsample', '.depthwise.', '.pointwise.']
    if all(x not in k for x in to_check):
        return True

def center_initialisation(new_value, new_shape, dimension_length):
    index_repeat = torch.nonzero(torch.tensor(new_shape) == dimension_length).item()
    new_value = new_value.unsqueeze(dim=index_repeat)
    pad_value = [[i//2, i//2] if idx == index_repeat else [0, 0] for idx, i in enumerate(new_shape)][::-1]
    if dimension_length % 2 == 0:
        idx = len(new_shape)-1-index_repeat
        pad_value[idx][1] -= 1
    pad_value = [t for j in pad_value for t in j]
    new_value = F.pad(new_value, pad_value)
    return new_value

def inflate_weights(v, new_shape, dimension_length):
    index_repeat = torch.nonzero(torch.tensor(new_shape) == dimension_length).item()
    nb_to_repeat = new_shape[index_repeat]//v.shape[index_repeat]
    new_value = torch.repeat_interleave(v, repeats=nb_to_repeat, dim=index_repeat) / nb_to_repeat
    return new_value

def handle_swin_contextual_tables(new_k, v, model_dict, new_dict, block_number, depth, starting_layer_number=None, dimension='2d', bottleneck=False):
    contextual_tables = replace_rpe_swin_contextual(new_k)
    for table in contextual_tables:
        new_shape = model_dict[table].shape
        new_value = v.permute(1, 0)
        new_value = center_initialisation(new_value, new_shape, 32)
        if dimension == '3d':
            new_value = inflate_weights(new_value, new_shape, 507)
        new_dict.update({table:new_value})
        if block_number == 1 and depth > 2:
            temp_list = table.split('.')
            for i in range(2, depth):
                temp_list[2] = str(i)
                new_layer_name = '.'.join(temp_list)
                new_dict.update({new_layer_name:new_value})
        if bottleneck is False:
            new_k_decoder = replace_vit_swin_decoder(table, starting_layer_number)
            new_dict.update({new_k_decoder:new_value})

def handle_swin_bias_tables(new_k, v, model_dict, new_dict, block_number, depth, starting_layer_number=None, dimension='2d', bottleneck=False):
    bias_table = new_k.replace('relative_position_bias_table', 'rpe_table')
    new_shape = model_dict[bias_table].shape
    new_value = v.permute(1, 0)
    if dimension == '3d':
        new_value = inflate_weights(new_value, new_shape, 507)
    new_dict.update({bias_table:new_value})
    if block_number == 1 and depth > 2:
        temp_list = bias_table.split('.')
        for i in range(2, depth):
            temp_list[2] = str(i)
            new_layer_name = '.'.join(temp_list)
            new_dict.update({new_layer_name:new_value})
    if bottleneck is False:
        new_k_decoder = replace_vit_swin_decoder(bias_table, starting_layer_number)
        new_dict.update({new_k_decoder:new_value})

def handle_swin_no_rpe_tables(new_k, v, new_dict, block_number, depth, starting_layer_number=None, bottleneck=False):
    new_dict.update({new_k:v})
    if block_number == 1 and depth > 2:
        temp_list = new_k.split('.')
        for i in range(2, depth):
            temp_list[2] = str(i)
            new_layer_name = '.'.join(temp_list)
            new_dict.update({new_layer_name:v})
    if bottleneck is False:
        new_k_decoder = replace_vit_swin_decoder(new_k, starting_layer_number)
        new_dict.update({new_k_decoder:v})

#def handle_bias_vit(v):
#    index_repeat = torch.nonzero(torch.tensor(new_shape) == dimension_length).item()

def load_weights_enc_dec_swin(model, pretrained_path_enc_dec, transformer_depth, device, rpe_mode, starting_layer_number):
    nb_blocks_per_layer = transformer_depth[0]
    pretrained_dict = torch.load(pretrained_path_enc_dec, map_location=device)
    pretrained_dict = pretrained_dict['model']
    model_dict = model.state_dict()
    new_dict = {}
    for k, v in pretrained_dict.items():
        if 'blocks' in k:
            block_number = int(k.split('.')[3])
            swin_layer_number = int(k.split('.')[1])
            if block_number > nb_blocks_per_layer - 1 or swin_layer_number > 2:
                continue
            new_k = replace_swin_encoder(k, swin_layer_number, block_number, starting_layer_nb=starting_layer_number)
            if 'relative_position_bias_table' in new_k and rpe_mode == 'contextual':
                handle_swin_contextual_tables(new_k, v, model_dict, new_dict, block_number, nb_blocks_per_layer, starting_layer_number=starting_layer_number)
            elif 'relative_position_bias_table' in new_k and rpe_mode == 'bias':
                handle_swin_bias_tables(new_k, v, model_dict, new_dict, block_number, nb_blocks_per_layer, starting_layer_number=starting_layer_number)
            else:
                handle_swin_no_rpe_tables(new_k, v, new_dict, block_number, nb_blocks_per_layer, starting_layer_number=starting_layer_number)
        elif k.startswith('norm.'):
            new_k = 'decoder.' + k
            new_dict.update({new_k:v})
    
    for k in list(new_dict.keys()):
        if k in model_dict:
            if new_dict[k].shape != model_dict[k].shape:
                print("delete:{};shape pretrain:{};shape model:{}".format(k,new_dict[k].shape,model_dict[k].shape))
                del new_dict[k]
    
    return new_dict

def load_weights_enc_dec_vit(model, pretrained_path_enc_dec, transformer_depth, device, starting_layer_number, rpe_mode, proj):
    nb_blocks_per_layer = transformer_depth[0]
    starting_layer = 2 if len(transformer_depth) == 3 else 3
    pretrained_dict = torch.load(pretrained_path_enc_dec, map_location=device)
    pretrained_dict = pretrained_dict['model']
    model_dict = model.state_dict()
    new_dict = {}
    for k, v in pretrained_dict.items():
        if 'blocks' in k:
            if check_conditions_vit(k, proj, rpe_mode):
                continue
            block_number = int(k.split('.')[1])
            if block_number > sum(transformer_depth) - 1:
                break
            new_k = replace_vit_encoder(k, starting_layer, block_number, nb_blocks_per_layer, rpe_mode)
            new_shape = model_dict[new_k].shape
            if rpe_mode == 'bias':
                v = v.squeeze()
            interpolation_mode = 'trilinear' if len(new_shape) == 3 else 'bilinear' if len(new_shape) == 2 else 'linear'
            new_value = F.interpolate(v.reshape((1, 1) + v.shape), new_shape, mode=interpolation_mode, align_corners=False).squeeze()
            new_dict.update({new_k:new_value})
            new_k_decoder = replace_vit_swin_decoder(new_k, starting_layer_number)
            new_dict.update({new_k_decoder:new_value})
    
    for k in list(new_dict.keys()):
        if k in model_dict:
            if new_dict[k].shape != model_dict[k].shape:
                print("delete:{};shape pretrain:{};shape model:{}".format(k,new_dict[k].shape,model_dict[k].shape))
                del new_dict[k]

    return new_dict


def load_swin_bottleneck(model, new_dict, pretrained_path_enc_dec, bottleneck_depth, device, rpe_mode, dimension):
    pretrained_dict = torch.load(pretrained_path_enc_dec, map_location=device)
    pretrained_dict = pretrained_dict['model']
    model_dict = model.state_dict()
    for k, v in pretrained_dict.items():
        if 'blocks' in k:
            block_number = int(k.split('.')[3])
            swin_layer_number = int(k.split('.')[1])
            if swin_layer_number != 3:
                continue
            new_k = '.'.join(['bottleneck', 'blocks', str(block_number)] + k.split('.')[4:])
            if 'relative_position_bias_table' in new_k and rpe_mode == 'contextual':
                handle_swin_contextual_tables(new_k, v, model_dict, new_dict, block_number, bottleneck_depth, dimension, bottleneck=True)
            elif 'relative_position_bias_table' in new_k and rpe_mode == 'bias':
                handle_swin_bias_tables(new_k, v, model_dict, new_dict, block_number, bottleneck_depth, dimension, bottleneck=True)
            else:
                handle_swin_no_rpe_tables(new_k, v, new_dict, block_number, bottleneck_depth, bottleneck=True)
    
    for k in list(new_dict.keys()):
        if k in model_dict:
            if new_dict[k].shape != model_dict[k].shape:
                print("delete:{};shape pretrain:{};shape model:{}".format(k,new_dict[k].shape,model_dict[k].shape))
                del new_dict[k]

    msg = model.load_state_dict(new_dict, strict=False)
    for x in msg[1]:
        if check_conditions_unexpected_swin(x):
            print(f'Unexpected keys: {x}')
    for y in msg[0]:
        if check_conditions_missing_swin(y):
            print(f'Missing keys: {y}')


def load_vit_bottleneck(model, new_dict, pretrained_path_enc_dec, transformer_depth, bottleneck_depth, device, rpe_mode, proj):
    pretrained_dict = torch.load(pretrained_path_enc_dec, map_location=device)
    pretrained_dict = pretrained_dict['model']
    model_dict = model.state_dict()
    for k, v in pretrained_dict.items():
        if 'blocks' in k:
            block_number = int(k.split('.')[1])
            if check_conditions_vit(k, proj, rpe_mode) or block_number < sum(transformer_depth):
                continue
            if block_number > sum(transformer_depth) + bottleneck_depth - 1:
                break
            new_k = replace_vit_bottleneck(k, block_number, bottleneck_depth, rpe_mode)
            new_shape = model_dict[new_k].shape
            if rpe_mode == 'bias':
                v = v.squeeze()
            interpolation_mode = 'trilinear' if len(new_shape) == 3 else 'bilinear' if len(new_shape) == 2 else 'linear'
            new_value = F.interpolate(v.reshape((1, 1,) + v.shape), new_shape, mode=interpolation_mode, align_corners=False).squeeze()
            new_dict.update({new_k:new_value})
    
    for k in list(new_dict.keys()):
        if k in model_dict:
            if new_dict[k].shape != model_dict[k].shape:
                print("delete:{};shape pretrain:{};shape model:{}".format(k,new_dict[k].shape,model_dict[k].shape))
                del new_dict[k]
    
    msg = model.load_state_dict(new_dict, strict=False)
    for x in msg[1]:
        print(f'Unexpected keys: {x}')
    for y in msg[0]:
        print(f'Missing keys: {y}')










































def load_swin_weights(model, pretrained_path, nb_bootleneck_block):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    pretrained_dict = pretrained_dict['model']
    model_dict = model.state_dict()
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('norm'):
            new_layer_name = 'encoder.' + k
            new_dict.update({new_layer_name:v})
        elif "layers." in k and 'blocks' in k:
            swin_layer_number = int(k[7:8])
            current_block_num = int(k[16:17])
            if current_block_num > 1:
                continue
            if swin_layer_number == 3 and nb_bootleneck_block > 0:
                new_layer_name = '.'.join(['bottleneck', 'blocks', str(current_block_num)] + k.split('.')[4:])
                #new_layer_name = 'bottleneck' + k[8:]
                new_dict.update({new_layer_name:v})
                if current_block_num == 1 and nb_bootleneck_block > 2:
                    temp_list = new_layer_name.split('.')
                    for i in range(2, nb_bootleneck_block):
                        temp_list[2] = str(i)
                        new_layer_name = '.'.join(temp_list)
                        new_dict.update({new_layer_name:v})
            else:
                current_decoder_layer_num = 2-swin_layer_number
                #current_encoder_layer_num = 2+swin_layer_number
                current_encoder_layer_num = swin_layer_number
                if current_decoder_layer_num < 0:
                    continue
                new_layer_name = '.'.join(['encoder', 'layers', str(current_encoder_layer_num), 'blocks', str(current_block_num)] + k.split('.')[4:])
                #new_layer_name = 'encoder.' + k
                new_dict.update({new_layer_name:v})
                #current_k = "decoder.layers_up." + str(current_decoder_layer_num) + k[8:]
                current_k = '.'.join(['decoder', 'layers', str(current_decoder_layer_num), 'blocks', str(current_block_num)] + k.split('.')[4:])
                new_dict.update({current_k:v})
    for k in list(new_dict.keys()):
        if k in model_dict:
            if new_dict[k].shape != model_dict[k].shape:
                print("delete:{};shape pretrain:{};shape model:{}".format(k,new_dict[k].shape,model_dict[k].shape))
                del new_dict[k]

    msg = model.load_state_dict(new_dict, strict=False)
    for x in msg[1]:
        print(f'Unexpected keys: {x}')
    for y in msg[0]:
        print(f'Missing keys: {y}')


#def load_weights_vit_bottleneck(model, pretrained_path):
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    pretrained_dict = torch.load(pretrained_path, map_location=device)
#    model_dict = model.state_dict()
#    full_dict = copy.deepcopy(model_dict)
#    model_start = [idx for idx, name in enumerate(model_dict.keys()) if 'temporal_block.encoder' in name][0]
#    model_end = [idx for idx, name in enumerate(model_dict.keys()) if 'mlp.layers' in name][0]
#    model_bus_layer_start = [idx for idx, name in enumerate(model_dict.keys()) if '.bus_layers.' in name][0]
#    vit_even_layer_name = [name for name in pretrained_dict.keys() if 'blocks.' in name and int(name.split('.')[1]) % 2 == 0]
#    vit_odd_layer_name = [name for name in pretrained_dict.keys() if 'blocks.' in name and int(name.split('.')[1]) % 2 != 0]
#    idx = 0
#    for model_key in list(model_dict.keys())[model_start:model_bus_layer_start]:
#        if 'lookup' in model_key:
#            continue
#        value = pretrained_dict[vit_even_layer_name[idx]]
#        value_shape = value.shape
#        if 3072 in value_shape:
#            new_shape = tuple([2048 if x == 3072 else x for x in value_shape])
#            value = F.interpolate(value.reshape((1, 1) + value_shape), new_shape, mode='bilinear' if len(value_shape) == 2 else 'linear', align_corners=False).squeeze()
#        full_dict.update({model_key:value})
#        idx += 1
#    for idx, model_key in enumerate(list(model_dict.keys())[model_bus_layer_start:model_end]):
#        value = pretrained_dict[vit_odd_layer_name[idx]]
#        value_shape = value.shape
#        if 3072 in value_shape:
#            new_shape = tuple([2048 if x == 3072 else x for x in value_shape])
#            value = F.interpolate(value.reshape((1, 1) + value_shape), new_shape, mode='bilinear' if len(value_shape) == 2 else 'linear', align_corners=False).squeeze()
#        full_dict.update({model_key:value})
#        
#    for k in list(full_dict.keys()):
#        if k in model_dict:
#            if full_dict[k].shape != model_dict[k].shape:
#                print("delete:{};shape pretrain:{};shape model:{}".format(k,full_dict[k].shape,model_dict[k].shape))
#                del full_dict[k]
#
#    msg = model.load_state_dict(full_dict, strict=False)
#    for x in msg[1]:
#        print(f'Unexpected keys: {x}')
#    for y in msg[0]:
#        print(f'Missing keys: {y}')


def replace_key_vit(key):
    x = key.replace('blocks', 'bottleneck_layers')
    x = x.replace('attn', 'self_attn')
    x = x.replace('mlp.fc1', 'linear1')
    x = x.replace('mlp.fc2', 'linear2')
    x = x.replace('qkv.weight', 'in_proj_weight')
    x = x.replace('qkv.bias', 'in_proj_bias')
    x = x.replace('proj.weight', 'out_proj.weight')
    x = x.replace('proj.bias', 'out_proj.bias')
    return x

def load_weights_irpe_bottleneck(model, pretrained_path, bottleneck_dim, n_heads, num_bottleneck_layers, bottleneck_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    pretrained_dict = pretrained_dict['model']
    model_dict = model.state_dict()
    full_dict = copy.deepcopy(model_dict)
    for k, v in pretrained_dict.items():
        if 'blocks' in k:
            if int(k[7]) >= num_bottleneck_layers:
                break
            if 'lookup' in k:
                new_shape = tuple([bottleneck_size**2 if x == 50 else bottleneck_dim//n_heads if x == 64 else n_heads for x in v.shape])
                v = F.interpolate(v.reshape((1, 1) + v.shape), new_shape, mode='trilinear', align_corners=False).squeeze()
            new_k = replace_key_vit(k)
            full_dict.update({new_k:v})
    for k in list(full_dict.keys()):
        if k in model_dict:
            if full_dict[k].shape != model_dict[k].shape:
                print("delete:{};shape pretrain:{};shape model:{}".format(k,full_dict[k].shape,model_dict[k].shape))
                del full_dict[k]

    msg = model.load_state_dict(full_dict, strict=False)
    for x in msg[1]:
        print(f'Unexpected keys: {x}')
    for y in msg[0]:
        print(f'Missing keys: {y}')


def load_weights_vit_bottleneck(model, pretrained_path, num_bottleneck_layers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    model_dict = model.state_dict()
    full_dict = copy.deepcopy(model_dict)
    for k, v in pretrained_dict.items():
        if 'blocks' in k:
            if int(k[7]) >= num_bottleneck_layers:
                break
            new_k = replace_key_vit(k)
            full_dict.update({new_k:v})
    for k in list(full_dict.keys()):
        if k in model_dict:
            if full_dict[k].shape != model_dict[k].shape:
                print("delete:{};shape pretrain:{};shape model:{}".format(k,full_dict[k].shape,model_dict[k].shape))
                del full_dict[k]

    msg = model.load_state_dict(full_dict, strict=False)
    for x in msg[1]:
        print(f'Unexpected keys: {x}')
    for y in msg[0]:
        print(f'Missing keys: {y}')



#def load_weights_irpe_bottleneck(model, pretrained_path, dim, n_heads):
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    pretrained_dict = torch.load(pretrained_path, map_location=device)
#    pretrained_dict = pretrained_dict['model']
#    model_dict = model.state_dict()
#    full_dict = copy.deepcopy(model_dict)
#    model_start = [idx for idx, name in enumerate(model_dict.keys()) if 'temporal_block.encoder' in name][0]
#    model_end = [idx for idx, name in enumerate(model_dict.keys()) if 'mlp.layers' in name][0]
#    model_bus_layer_start = [idx for idx, name in enumerate(model_dict.keys()) if '.bus_layers.' in name][0]
#    vit_even_layer_name = [name for name in pretrained_dict.keys() if 'blocks.' in name and int(name.split('.')[1]) % 2 == 0]
#    vit_odd_layer_name = [name for name in pretrained_dict.keys() if 'blocks.' in name and int(name.split('.')[1]) % 2 != 0]
#    for idx, model_key in enumerate(list(model_dict.keys())[model_start:model_bus_layer_start]):
#        value = pretrained_dict[vit_even_layer_name[idx]]
#        value_shape = value.shape
#        if 3072 in value_shape:
#            new_shape = tuple([2048 if x == 3072 else x for x in value_shape])
#            value = F.interpolate(value.reshape((1, 1) + value_shape), new_shape, mode='bilinear' if len(value_shape) == 2 else 'linear', align_corners=False).squeeze()
#        if 'lookup' in model_key:
#            new_shape = tuple([57 if x == 50 else dim if x == 64 else n_heads for x in value_shape])
#            value = F.interpolate(value.reshape((1, 1) + value_shape), new_shape, mode='trilinear', align_corners=False).squeeze()
#        full_dict.update({model_key:value})
#    idx = 0
#    for model_key in list(model_dict.keys())[model_bus_layer_start:model_end]:
#        vit_layer_name = vit_odd_layer_name[idx]
#        while 'lookup' in vit_layer_name:
#            idx += 1
#            vit_layer_name = vit_odd_layer_name[idx]
#        value = pretrained_dict[vit_layer_name]
#        value_shape = value.shape
#        if 3072 in value_shape:
#            new_shape = tuple([2048 if x == 3072 else x for x in value_shape])
#            value = F.interpolate(value.reshape((1, 1) + value_shape), new_shape, mode='bilinear' if len(value_shape) == 2 else 'linear', align_corners=False).squeeze()
#        full_dict.update({model_key:value})
#        idx += 1
#        
#    for k in list(full_dict.keys()):
#        if k in model_dict:
#            if full_dict[k].shape != model_dict[k].shape:
#                print("delete:{};shape pretrain:{};shape model:{}".format(k,full_dict[k].shape,model_dict[k].shape))
#                del full_dict[k]
#
#    msg = model.load_state_dict(full_dict, strict=False)
#    for x in msg[1]:
#        print(f'Unexpected keys: {x}')
#    for y in msg[0]:
#        print(f'Missing keys: {y}')


def replace_key_rpe_temporal(key, rpe_mode, block_number, bottleneck_depth):
    new_block_number = block_number % bottleneck_depth
    x = '.'.join(['bottleneck', 'encoder', 'enc_layers', str(new_block_number)] + key.split('.')[2:])
    x = x.replace('attn', 'self_attn')
    if rpe_mode == 'contextual':
        x = x.replace('rpe_q.lookup_table_weight', 'q_rpe_table')
        x = x.replace('rpe_k.lookup_table_weight', 'k_rpe_table')
        x = x.replace('rpe_v.lookup_table_weight', 'v_rpe_table')
    elif rpe_mode == 'bias':
        x = x.replace('rpe_q.lookup_table_weight', 'rpe_table')
        x = x.replace('rpe_k.lookup_table_weight', 'rpe_table')
        x = x.replace('rpe_v.lookup_table_weight', 'rpe_table')
    
    y = x.replace('enc_layers', 'bus_layers')

    return x, y


def load_weights_ifc_bottleneck(model, new_dict, pretrained_path_ifc, pretrained_path_irpe, rpe_mode, bottleneck_depth, transformer_depth, dim_feedforward, proj):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict_ifc = torch.load(pretrained_path_ifc, map_location=device)
    pretrained_dict_irpe = torch.load(pretrained_path_irpe, map_location=device)
    pretrained_dict_ifc = pretrained_dict_ifc['model']
    pretrained_dict_irpe = pretrained_dict_irpe['model']
    model_dict = model.state_dict()
    for k, v in pretrained_dict_irpe.items():
        if 'blocks' in k:
            block_number = int(k.split('.')[1])
            if check_conditions(k, proj, rpe_mode) or block_number < sum(transformer_depth):
                continue
            if block_number > sum(transformer_depth) + bottleneck_depth - 1:
                break
            if 'lookup' in k:
                new_k_spatial, new_k_temporal = replace_key_rpe_temporal(k, rpe_mode, block_number, bottleneck_depth)
                new_shape_spatial = model_dict[new_k_spatial].shape
                new_shape_temporal = model_dict[new_k_temporal].shape
                if rpe_mode == 'contextual':
                    value_spatial = F.interpolate(v.reshape((1, 1) + v.shape), new_shape_spatial, mode='trilinear', align_corners=False).squeeze()
                    value_temporal = F.interpolate(v.reshape((1, 1) + v.shape), new_shape_temporal, mode='trilinear', align_corners=False).squeeze()
                elif rpe_mode == 'bias':
                    v = v[None, :, :, :]
                    value_spatial = F.interpolate(v, new_shape_spatial, mode='bilinear', align_corners=False).squeeze()
                    value_temporal = F.interpolate(v, new_shape_temporal, mode='bilinear', align_corners=False).squeeze()
                new_dict.update({new_k_spatial:value_spatial})
                new_dict.update({new_k_temporal:value_temporal})
    for k, v in pretrained_dict_ifc.items():
        if 'transformer.memory' in k:
            new_k = k.replace('detr.transformer', 'bottleneck')
            new_shape = model_dict[new_k].shape
            value = F.interpolate(v.reshape((1, 1) + v.shape), new_shape, mode='bilinear', align_corners=False).squeeze() # Change for row wise
            new_dict.update({new_k:value})
        elif 'transformer.encoder' in k:
            new_k = k.replace('detr.transformer', 'bottleneck')
            new_shape = tuple([dim_feedforward if x == 2048 else x * 3 for x in v.shape])
            value = F.interpolate(v.reshape((1, 1) + v.shape), new_shape, mode='bilinear' if len(v.shape) == 2 else 'linear', align_corners=False).squeeze()
            new_dict.update({new_k:value})
        
    for k in list(new_dict.keys()):
        if k in model_dict:
            if new_dict[k].shape != model_dict[k].shape:
                print("delete:{};shape pretrain:{};shape model:{}".format(k,new_dict[k].shape,model_dict[k].shape))
                del new_dict[k]

    msg = model.load_state_dict(new_dict, strict=False)
    for x in msg[1]:
        print(f'Unexpected keys: {x}')
    for y in msg[0]:
        print(f'Missing keys: {y}')




#def load_weights_ifc_bottleneck(model, pretrained_path):
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    pretrained_dict = torch.load(pretrained_path, map_location=device)
#    pretrained_dict = pretrained_dict['model']
#    model_dict = model.state_dict()
#    full_dict = copy.deepcopy(model_dict)
#    for k, v in pretrained_dict.items():
#        print(k)
#        if 'transformer.memory' in k:
#            k_split = '.'.join(k.split('.')[2:])
#            new_layer_name = 'temporal_block.' + k_split
#            new_shape = (v.shape[0], v.shape[1] * 3)
#            # Change for row wise
#            value = F.interpolate(v.reshape((1, 1) + v.shape), new_shape, mode='bilinear', align_corners=False).squeeze() # Change for row wise
#            full_dict.update({new_layer_name:value})
#        elif 'transformer.encoder' in k:
#            k_split = '.'.join(k.split('.')[2:])
#            new_layer_name = 'temporal_block.' + k_split
#            # Change for row wise
#            new_shape = tuple([x * 3 if x != 2048 else x for x in v.shape])
#            value = F.interpolate(v.reshape((1, 1) + v.shape), new_shape, mode='bilinear' if len(v.shape) == 2 else 'linear', align_corners=False).squeeze()
#            full_dict.update({new_layer_name:value})
#    for k in list(full_dict.keys()):
#        if k in model_dict:
#            if full_dict[k].shape != model_dict[k].shape:
#                print("delete:{};shape pretrain:{};shape model:{}".format(k,full_dict[k].shape,model_dict[k].shape))
#                del full_dict[k]
#
#    msg = model.load_state_dict(full_dict, strict=False)
#    for x in msg[1]:
#        print(f'Unexpected keys: {x}')
#    for y in msg[0]:
#        print(f'Missing keys: {y}')