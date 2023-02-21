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
                current_encoder_layer_num = 2+swin_layer_number
                new_layer_name = '.'.join(['encoder', 'layers', str(current_encoder_layer_num), 'blocks', str(current_block_num)] + k.split('.')[4:])
                #new_layer_name = 'encoder.' + k
                new_dict.update({new_layer_name:v})
                current_k = '.'.join(['decoder', 'layers', str(current_decoder_layer_num), 'blocks', str(current_block_num)] + k.split('.')[4:])
                #current_k = "decoder.layers_up." + str(current_decoder_layer_num) + k[8:]
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