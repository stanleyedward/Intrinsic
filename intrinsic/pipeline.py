import torch
import numpy as np
from skimage.transform import resize

from chrislib.resolution_util import optimal_resize
from chrislib.general import round_32, uninvert, invert, get_brightness, to2np
from chrislib.color_util import batch_rgb2iuv, batch_iuv2rgb

from intrinsic.ordinal_util import base_resize, equalize_predictions

from altered_midas.midas_net import MidasNet
from altered_midas.midas_net_custom import MidasNet_small

STAGE_DICT = {
    # 'ordinal': 0,
    'gray': 1,
    'chroma': 2,
    'albedo': 3,
    'diffuse': 4
}

V1_DICT = {
    'paper_weights' : 'https://github.com/compphoto/Intrinsic/releases/download/v1.0/final_weights.pt',
    'rendered_only' : 'https://github.com/compphoto/Intrinsic/releases/download/v1.0/rendered_only_weights.pt'
}

def load_decompile(path):
    """call torch.load then convert the loaded keys from torch.compile to regular"""
    compiled_dict = torch.load(path)

    remove_prefix = '_orig_mod.'
    model_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in compiled_dict.items()}

    return model_dict
    
def load_models(path, stage=4, device='cuda', compiled=False, chroma_dpt=False, alb_residual=False):
    """The networks as part of our intrinsic decomposition pipeline. Since the pipeline consists of stages,
    can load the models up to a specific stage in the pipeline

    params:
        paths (str or list): the paths to each of the models in the pipeline, or a name for released weights
        stage (int or str) optional: the stage to load the models up to (1-4) (default 4)
            if string must be one of the following: "gray", "chroma", "albedo", "diffuse"
        device (str) optional: the device to run the model on (default "cuda")

    returns:
        models (dict): a dict with the following structure: {
            "ordinal_model": altered_midas.midas_net.MidasNet,
            "iid_model": altered_midas.midas_net_custom.MidasNet_small,
            "col_model": altered_midas.midas_net_custom.MidasNet,
            "alb_model": altered_midas.midas_net_custom.MidasNet,
            "dif_model": altered_midas.midas_net_custom.MidasNet
        }
    """
    models = {}

    load_func = load_decompile if compiled else torch.load
    
    if isinstance(stage, str):
        stage = STAGE_DICT[stage]

    # if the path is a string, we are loading a release of the pipeline
    if isinstance(path, str):
        if path in ['paper_weights', 'rendered_only']:
            # these are V1 releases from the ordinal shading paper, so set the stage to 1 to only run grayscale
            combined_dict = torch.hub.load_state_dict_from_url(V1_DICT[path], map_location=device, progress=True)
            stage = 1

            ord_state_dict = combined_dict['ord_state_dict']
            iid_state_dict = combined_dict['iid_state_dict']
        elif path == 'v2':
            print("loading v2 weights")
            base_url = 'https://github.com/compphoto/Intrinsic/releases/download/v2.0/'
            ord_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_0.pt' , map_location=device, progress=True)
            iid_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_1.pt' , map_location=device, progress=True)
            col_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_2.pt' , map_location=device, progress=True)
            alb_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_3.pt' , map_location=device, progress=True)
            dif_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_4.pt' , map_location=device, progress=True)
        elif path == 'v2.1':
            print("loading v2.1 weights")
            base_url = 'https://github.com/compphoto/Intrinsic/releases/download/v2.1/'
            ord_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_0_v21.pt' , map_location=device, progress=True)
            iid_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_1_v21.pt' , map_location=device, progress=True)
            col_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_2_v21.pt' , map_location=device, progress=True)
            alb_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_3_v21.pt' , map_location=device, progress=True)
            dif_state_dict = torch.hub.load_state_dict_from_url(base_url + 'stage_4_v21.pt' , map_location=device, progress=True)

            alb_residual = True

    elif isinstance(path, list):

        ord_state_dict = load_func(path[0])
        iid_state_dict = load_func(path[1])

        if stage >= 2: col_state_dict = load_func(path[2])
        if stage >= 3: alb_state_dict = load_func(path[3])
        if stage >= 4: dif_state_dict = load_func(path[4])

    ord_model = MidasNet()
    ord_model.load_state_dict(ord_state_dict)
    ord_model.eval()
    ord_model = ord_model.to(device)
    models['ord_model'] = ord_model

    iid_model = MidasNet_small(exportable=False, input_channels=5, output_channels=1)
    iid_model.load_state_dict(iid_state_dict)
    iid_model.eval()
    iid_model = iid_model.to(device)
    models['iid_model'] = iid_model
    
    if stage >= 2:
        if chroma_dpt:
            col_model = DPTDepthModel(in_chan=7, out_chan=2)
        else:
            col_model = MidasNet(activation='sigmoid', input_channels=7, output_channels=2)

        col_model.load_state_dict(col_state_dict)
        col_model.eval()
        col_model = col_model.to(device)
        models['col_model'] = col_model
    
    if stage >= 3:
        alb_model = MidasNet(activation='sigmoid', input_channels=9, output_channels=3, last_residual=alb_residual)
        alb_model.load_state_dict(alb_state_dict)
        alb_model.eval()
        alb_model = alb_model.to(device)
        models['alb_model'] = alb_model
    
    if stage >= 4:
        dif_model = MidasNet(activation='sigmoid', input_channels=9, output_channels=3)
        dif_model.load_state_dict(dif_state_dict)
        dif_model.eval()
        dif_model = dif_model.to(device)
        models['dif_model'] = dif_model

    return models

def run_gray_pipeline(
        models,
        img_arr,
        resize_conf=0.0,
        base_size=384,
        maintain_size=False,
        linear=False,
        device='cuda',
        lstsq_p=0.0,
        inputs='all'):
    """Runs the complete pipeline for grayscale shading and albedo prediction

    params:
        models (dict): models dictionary returned by load_models()
        img_arr (np.array): RGB input image as numpy array between 0-1
        resize_conf (float) optional: confidence to use for resizing (between 0-1) if None maintain
            original size (default None)
        base_size (int) optional: size of the base resolution estimation (default 384)
        maintain_size (bool) optional: whether or not the results match the input image size
            (default False)
        linear (bool) optional: whether or not the input image is already linear (default False)
        device (str) optional: string representing device to use for pipeline (default "cuda")
        lstsq_p (float) optional: subsampling factor for computing least-squares fit 
            when matching the scale of base and full estimations (default 0.0)
        inputs (str) optional: network inputs ("full", "base", "rgb", "all") the rgb image is
            always included (default "all")

    returns:
        results (dict): a result dictionary with albedo, shading and potentially ordinal estimations
    """
    results = {}

    orig_h, orig_w, _ = img_arr.shape
    
    # if no confidence value set, just round original size to 32 for model input
    if resize_conf is None:
        img_arr = resize(img_arr, (round_32(orig_h), round_32(orig_w)), anti_aliasing=True)

    # if a the confidence is an int, just rescale image so that the large side
    # of the image matches the specified integer value
    elif isinstance(resize_conf, int):
        scale = resize_conf / max(orig_h, orig_w)
        img_arr = resize(
            img_arr,
            (round_32(orig_h * scale), round_32(orig_w * scale)),
            anti_aliasing=True)
    
    # if the confidence is a float use the optimal resize code from Miangoleh et al.
    elif isinstance(resize_conf, float):
        img_arr = optimal_resize(img_arr, conf=resize_conf)

    fh, fw, _ = img_arr.shape
    
    # if the image is in sRGB we do simple linearization using gamma=2.2
    if not linear:
        lin_img = img_arr ** 2.2
    else:
        lin_img = img_arr

    with torch.no_grad():
        # ordinal shading estimation --------------------------
        
        # resize image for base and full estimations and send through ordinal net
        base_input = base_resize(lin_img, base_size)
        full_input = lin_img

        base_input = torch.from_numpy(base_input).permute(2, 0, 1).to(device).float()
        full_input = torch.from_numpy(full_input).permute(2, 0, 1).to(device).float()

        base_out = models['ord_model'](base_input.unsqueeze(0)).squeeze(0)
        full_out = models['ord_model'](full_input.unsqueeze(0)).squeeze(0)
        
        # the ordinal estimations come out of the model with a channel dim
        base_out = base_out.permute(1, 2, 0).cpu().numpy()
        full_out = full_out.permute(1, 2, 0).cpu().numpy()

        base_out = resize(base_out, (fh, fw))

        # if we are using all inputs, we scale the input estimations using the base estimate
        if inputs == 'all':
            ord_base, ord_full = equalize_predictions(lin_img, base_out, full_out, p=lstsq_p)
        else:
            ord_base, ord_full = base_out, full_out
        # ------------------------------------------------------

        # ordinal shading to real shading ----------------------
        inp = torch.from_numpy(lin_img).permute(2, 0, 1).to(device)
        bse = torch.from_numpy(ord_base).permute(2, 0, 1).to(device)
        fll = torch.from_numpy(ord_full).permute(2, 0, 1).to(device)
        
        # combine the base and full ordinal estimations w/ the input image
        # NOTE: this is just for ablation studies provided in the paper
        if inputs == 'full':
            combined = torch.cat((inp, fll), 0).unsqueeze(0)
        elif inputs == 'base':
            combined = torch.cat((inp, bse), 0).unsqueeze(0)
        elif inputs == 'rgb':
            combined = inp.unsqueeze(0)
        else:
            combined = torch.cat((inp, bse, fll), 0).unsqueeze(0)

        inv_shd = models['iid_model'](combined).squeeze(1)
        
        # the shading comes out in the inverse space so undo it 
        shd = uninvert(inv_shd)
        alb = inp / shd
        # ------------------------------------------------------
    
    # put all the outputs into a dictionary to return
    inv_shd = inv_shd.squeeze(0).detach().cpu().numpy()
    alb = alb.permute(1, 2, 0).detach().cpu().numpy()

    if maintain_size:
        ord_base = resize(base_out, (orig_h, orig_w), anti_aliasing=True)
        ord_full = resize(full_out, (orig_h, orig_w), anti_aliasing=True)

        inv_shd = resize(inv_shd, (orig_h, orig_w), anti_aliasing=True)
        alb = resize(alb, (orig_h, orig_w), anti_aliasing=True)


    results['ord_full'] = ord_full
    results['ord_base'] = ord_base

    results['gry_shd'] = inv_shd
    results['gry_alb'] = alb
    results['image'] = img_arr
    results['lin_img'] = lin_img

    return results

def run_pipeline(models, img_arr, stage=4, resize_conf=0.0, base_size=384, linear=False, device='cuda', specular_threshold=0.05, inputs='all', maintain_size=False):
    
    results = run_gray_pipeline(
        models,
        img_arr,
        resize_conf=resize_conf,
        linear=linear,
        device=device,
        base_size=base_size,
        inputs=inputs,
        maintain_size=maintain_size
    )

    if stage == 1:
        return results

    img = results['lin_img']
    gry_shd = results['gry_shd'][:, :, None]
    gry_alb = results['gry_alb']

    # pytorch versions of the input, gry shd and albedo with channel dim
    net_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    net_shd = torch.from_numpy(gry_shd).permute(2, 0, 1).unsqueeze(0).to(device)
    net_alb = torch.from_numpy(gry_alb).permute(2, 0, 1).unsqueeze(0).to(device)

    in_img_luv = batch_rgb2iuv(net_img)
    in_alb_luv = batch_rgb2iuv(net_alb)

    orig_sz = img.shape[:2]
    scale = base_size / max(orig_sz)
    base_sz = (round_32(orig_sz[0] * scale), round_32(orig_sz[1] * scale))

    # we want to resize the inputs to base resolution
    in_img_luv = torch.nn.functional.interpolate(in_img_luv, size=base_sz, mode='bilinear', align_corners=True, antialias=True)
    in_alb_luv = torch.nn.functional.interpolate(in_alb_luv, size=base_sz, mode='bilinear', align_corners=True, antialias=True)
    in_gry_shd = torch.nn.functional.interpolate(net_shd, size=base_sz, mode='bilinear', align_corners=True, antialias=True)

    inp = torch.cat([in_img_luv, in_gry_shd, in_alb_luv], 1)

    # this is the shading color components, N x 2 x H x W
    with torch.no_grad():
        uv_shd = models['col_model'](inp)

    # resize the low res shd chroma back to original size
    uv_shd = torch.nn.functional.interpolate(uv_shd, size=orig_sz, mode='bilinear', align_corners=True)

    # now combine gry shd with chroma in channel dim and convert to rgb
    iuv_shd = torch.cat((net_shd, uv_shd), 1)
    rough_shd = batch_iuv2rgb(iuv_shd)
    rough_alb = net_img / rough_shd
    
    rough_alb *= 0.75 / torch.quantile(rough_alb, 0.99)
    rough_alb = rough_alb.clip(0.001)
    rough_shd = net_img / rough_alb

    # convert the low-res chroma decomposition to numpy in case we return early
    lr_clr = to2np(batch_iuv2rgb(torch.cat((torch.ones_like(net_shd) * 0.6, uv_shd), 1)).squeeze(0))
    lr_alb = to2np(rough_alb.squeeze(0))
    lr_shd = to2np(rough_shd.squeeze(0))
    wb_img = (lr_alb * get_brightness(lr_shd)).clip(0, 1)

    results['iuv_shd'] = to2np(iuv_shd.squeeze(0))
    results['lr_clr'] = lr_clr
    results['lr_alb'] = lr_alb
    results['lr_shd'] = lr_shd
    results['wb_img'] = wb_img

    if stage == 2:
        return results

    # albedo estimation net gets img, inverted rgb shd and implied alb
    inp = torch.cat([net_img, invert(rough_shd), rough_alb], 1)

    with torch.no_grad():
        pred_alb = models['alb_model'](inp)

    net_clr_shd = net_img / pred_alb.clip(1e-3)

    # convert high-res albedo and shading to numpy
    hr_alb = pred_alb.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    hr_shd = img / hr_alb.clip(1e-3)
    hr_clr = batch_rgb2iuv(net_img / pred_alb.clip(1e-4))
    hr_clr[:, 0, :, :] = torch.ones_like(net_shd) * 0.6
    hr_clr = to2np(batch_iuv2rgb(hr_clr).squeeze(0))
    wb_img = (hr_alb * get_brightness(hr_shd)).clip(0, 1)

    results['hr_alb'] = hr_alb
    results['hr_shd'] = hr_shd
    results['hr_clr'] = hr_clr
    results['wb_img'] = wb_img

    if stage == 3:
        return results

    inp = torch.cat([net_img, invert(net_clr_shd), pred_alb], 1)

    with torch.no_grad():
        dif_shd = models['dif_model'](inp)

    dif_shd = uninvert(dif_shd)

    dif_shd = dif_shd.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    dif_img = (hr_alb * dif_shd)
    res = img - dif_img
    
    neg_res = res.copy()
    neg_res[neg_res > 0] = 0
    neg_res = abs(neg_res)

    pos_res = res.copy()
    pos_res[pos_res < specular_threshold] = 0

    results['dif_shd'] = dif_shd
    results['dif_img'] = dif_img
    results['residual'] = res
    results['neg_res'] = neg_res
    results['pos_res'] = pos_res

    return results

def run_gray_pipeline_gpu(
        models,
        img_arr,
        resize_conf=0.0,
        base_size=384,
        maintain_size=False,
        linear=False,
        device='cuda',
        lstsq_p=0.0,
        inputs='all'):

    assert isinstance(img_arr, torch.Tensor), "img_arr must be a torch.Tensor"
    img = img_arr
    if img.device != torch.device(device):
        img = img.to(device)
    if img.dim() != 3:
        raise ValueError("img tensor must be 3D (H,W,3) or (3,H,W)")
    if img.shape[0] == 3 and img.shape[-1] != 3:  # CHW -> HWC
        img_hwc = img.permute(1, 2, 0)
    elif img.shape[-1] == 3:
        img_hwc = img
    else:
        raise ValueError("Could not infer channel dimension (expect 3 channels)")
    orig_h, orig_w = img_hwc.shape[:2]

    def torch_resize(hwc_tensor, target_hw):
        th, tw = target_hw
        chw = hwc_tensor.permute(2, 0, 1).unsqueeze(0)
        chw = torch.nn.functional.interpolate(
            chw, size=(th, tw), mode='bilinear', align_corners=True, antialias=True)
        return chw.squeeze(0).permute(1, 2, 0)

    if resize_conf is None:
        img_hwc = torch_resize(img_hwc, (round_32(orig_h), round_32(orig_w)))
    elif isinstance(resize_conf, int):
        scale = resize_conf / max(orig_h, orig_w)
        img_hwc = torch_resize(
            img_hwc,
            (round_32(int(orig_h * scale)), round_32(int(orig_w * scale))))
    elif isinstance(resize_conf, float):
        #TODO turn this to GPU only resize
        img_np = img_hwc.detach().cpu().numpy()
        img_np = optimal_resize(img_np, conf=resize_conf)
        img_hwc = torch.from_numpy(img_np).to(device)

    fh, fw = img_hwc.shape[:2]

    if not linear:
        lin_img = img_hwc ** 2.2
    else:
        lin_img = img_hwc

    with torch.no_grad():
        def torch_base_resize(hwc_tensor, base_sz):
            h, w = hwc_tensor.shape[:2]
            scale = base_sz / max(h, w)
            nh = round_32(int(h * scale))
            nw = round_32(int(w * scale))
            return torch_resize(hwc_tensor, (nh, nw))

        base_input = torch_base_resize(lin_img, base_size)
        full_input = lin_img

        # To CHW
        base_chw = base_input.permute(2, 0, 1).float()
        full_chw = full_input.permute(2, 0, 1).float()

        base_out = models['ord_model'](base_chw.unsqueeze(0)).squeeze(0)  # C,H,W
        full_out = models['ord_model'](full_chw.unsqueeze(0)).squeeze(0)

        base_out_hwc = base_out.permute(1, 2, 0)
        full_out_hwc = full_out.permute(1, 2, 0)

        if base_out_hwc.shape[:2] != (fh, fw):
            base_out_hwc = torch_resize(base_out_hwc, (fh, fw))

        if inputs == 'all':
            # port equalize_preds?
            base_np = base_out_hwc.detach().cpu().numpy()
            full_np = full_out_hwc.detach().cpu().numpy()
            lin_np = lin_img.detach().cpu().numpy()
            ord_base_np, ord_full_np = equalize_predictions(lin_np, base_np, full_np, p=lstsq_p)
            ord_base = torch.from_numpy(ord_base_np).to(device)
            ord_full = torch.from_numpy(ord_full_np).to(device)
        else:
            ord_base = base_out_hwc
            ord_full = full_out_hwc

        inp = lin_img.permute(2, 0, 1)                  # 3,H,W
        bse = ord_base.permute(2, 0, 1)
        fll = ord_full.permute(2, 0, 1)

        if inputs == 'full':
            combined = torch.cat((inp, fll), 0).unsqueeze(0)
        elif inputs == 'base':
            combined = torch.cat((inp, bse), 0).unsqueeze(0)
        elif inputs == 'rgb':
            combined = inp.unsqueeze(0)
        else:
            combined = torch.cat((inp, bse, fll), 0).unsqueeze(0)

        inv_shd = models['iid_model'](combined).squeeze(1)  # 1,H,W
        shd = uninvert(inv_shd)                             # 1,H,W
        alb = (inp / shd).clamp_min(1e-6)                   # 3,H,W

    inv_shd = inv_shd.squeeze(0)        # H,W
    gry_alb = alb.permute(1, 2, 0)      # H,W,3

    if maintain_size and (fh != orig_h or fw != orig_w):
        inv_shd = torch_resize(inv_shd.unsqueeze(-1), (orig_h, orig_w)).squeeze(-1)
        gry_alb = torch_resize(gry_alb, (orig_h, orig_w))
        ord_base = torch_resize(ord_base, (orig_h, orig_w))
        ord_full = torch_resize(ord_full, (orig_h, orig_w))
        lin_img = torch_resize(lin_img, (orig_h, orig_w))
        img_hwc = torch_resize(img_hwc, (orig_h, orig_w))

    results = {
        'ord_full': ord_full,      # H,W,C
        'ord_base': ord_base,
        'gry_shd': inv_shd,        # H,W
        'gry_alb': gry_alb,        # H,W,3
        'image': img_hwc,          # possibly resized image H,W,3
        'lin_img': lin_img         # linearized image H,W,3
    }
    return results

def run_residual_pipeline_gpu(
        models,
        img_arr,
        resize_conf=0.0,
        base_size=384,
        linear=False,
        device='cuda',
        specular_threshold=0.05,
        inputs='all'):

    gray = run_gray_pipeline_gpu(
        models,
        img_arr,
        resize_conf=resize_conf,
        base_size=base_size,
        linear=linear,
        device=device,
        inputs=inputs
    )
    img = gray['lin_img']                 # H,W,3 torch
    gry_shd = gray['gry_shd']             # H,W
    gry_alb = gray['gry_alb']             # H,W,3

    # Prep (to CHW)
    net_img = img.permute(2, 0, 1).unsqueeze(0)        # 1,3,H,W
    net_shd = gry_shd.unsqueeze(0).unsqueeze(0)        # 1,1,H,W
    net_alb = gry_alb.permute(2, 0, 1).unsqueeze(0)    # 1,3,H,W

    with torch.no_grad():
        img_iuv = batch_rgb2iuv(net_img)
        alb_iuv = batch_rgb2iuv(net_alb)

        orig_h, orig_w = img.shape[:2]
        scale = base_size / max(orig_h, orig_w)
        base_sz = (round_32(int(orig_h * scale)), round_32(int(orig_w * scale)))

        def interp(t, size):
            return torch.nn.functional.interpolate(
                t, size=size, mode='bilinear', align_corners=True, antialias=True)

        img_iuv_b = interp(img_iuv, base_sz)
        alb_iuv_b = interp(alb_iuv, base_sz)
        gry_shd_b = interp(net_shd, base_sz)

        col_inp = torch.cat([img_iuv_b, gry_shd_b, alb_iuv_b], 1)
        uv_shd = models['col_model'](col_inp)
        uv_shd = interp(uv_shd, (orig_h, orig_w))

        iuv_shd = torch.cat((net_shd, uv_shd), 1)
        rough_shd = batch_iuv2rgb(iuv_shd)             # 1,3,H,W
        rough_alb = (net_img / rough_shd).clamp(1e-3)

        rough_alb *= 0.75 / torch.quantile(rough_alb, 0.99)
        rough_alb = rough_alb.clamp(1e-3)
        rough_shd = net_img / rough_alb

        alb_inp = torch.cat([net_img, invert(rough_shd), rough_alb], 1)
        pred_alb = models['alb_model'](alb_inp).clamp(1e-3)

        net_clr_shd = net_img / pred_alb.clamp(1e-3)

        dif_inp = torch.cat([net_img, invert(net_clr_shd), pred_alb], 1)
        dif_shd_inv = models['dif_model'](dif_inp)
        dif_shd = uninvert(dif_shd_inv).clamp(1e-5)     # 1,3,H,W

        hr_alb = pred_alb.squeeze(0).permute(1, 2, 0)          # H,W,3
        dif_shd_hw3 = dif_shd.squeeze(0).permute(1, 2, 0)      # H,W,3
        dif_img = hr_alb * dif_shd_hw3
        res = img - dif_img

        # Residual separation
        neg_res = (-torch.minimum(res, torch.zeros_like(res)))  
        pos_res = res.clone()
        pos_res[pos_res < specular_threshold] = 0.0

    return {
        'pos_res': pos_res,              # H,W,3
        'neg_res': neg_res,              # H,W,3
        'residual': res,                 # H,W,3
        'dif_img': dif_img,              # H,W,3
        'diffuse_shading': dif_shd_hw3,  # H,W,3
        'albedo': hr_alb                 # H,W,3
    }