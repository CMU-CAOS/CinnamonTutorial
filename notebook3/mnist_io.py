import pickle

Primes = [204865537, 205651969, 206307329, 207880193, 209059841, 210370561, 211025921, 211812353, 214171649, 215482369, 215744513, 216137729, 216924161, 217317377, 218628097, 219676673, 220594177, 221249537, 222035969, 222167041, 222953473, 223215617, 224002049, 224133121, 225574913, 228065281, 228458497, 228720641, 230424577, 230686721, 230817793, 231473153, 232390657, 232652801, 234356737, 235798529, 236584961, 236716033, 239337473, 239861761, 240648193, 241827841, 244842497, 244973569, 245235713, 245760001, 246415361, 249561089, 253100033, 253493249, 254279681, 256376833, 256770049, 257949697, 258605057, 260571137, 260702209, 261488641, 261881857, 263323649, 263454721, 264634369, 265420801, 268042241]
class ValueMeta:
    def __init__(self, scale, level):
        self.scale = scale
        self.level = level
        if self.level <= 0:
            raise ValueError("Level must be positive")
        
    def rescale(self):
        return ValueMeta(self.scale/Primes[self.level - 1],self.level-1)

    def __mul__(self,other):
        if not isinstance(other,ValueMeta):
            raise ValueError("Must multiply by ValueMeta")
        if self.level != other.level:
            raise ValueError("Levels must be the same for Multiplication")
        return ValueMeta(self.scale*other.scale,self.level)

    def __add__(self,other):
        if not isinstance(other,ValueMeta):
            raise ValueError("Must multiply by ValueMeta")
        if self.level != other.level:
            raise ValueError("Levels must be the same for Addition")
        if self.scale != other.other:
            raise ValueError("Scales must be the same for Addition")
        return ValueMeta(self.scale,self.level)

    def __lsfhift__(self,_):
        return ValueMeta(self.scale,self.level)

    def __rsfhift__(self,_):
        return ValueMeta(self.scale,self.level)



ReferenceOutputs = {}

import numpy as np
SLOTS = 32*1024

def rotate(x,n):
    return np.concat((x[n:],x[:n]))

def pack_image(input_image,kernel_weights,stride):
    input_height, input_width = input_image.shape
    out_channels,  kernel_height, kernel_width = kernel_weights.shape
    out_height = (input_height - kernel_height) // stride + 1
    out_width = (input_width - kernel_width) // stride + 1
    im2col = np.zeros((out_height * out_width, kernel_height*kernel_width))
    im2col_packed = np.zeros(SLOTS)
    for h in range(out_height):
        for w in range(out_width):
            h_start = h * stride
            w_start = w * stride
            h_end = h_start + kernel_height
            w_end = w_start + kernel_width
            region = input_image[h_start:h_end, w_start:w_end]
            im2col [(h*out_width + w), :kernel_height * kernel_width] = region.reshape(1,-1)

    for i in range(64):
        for j in range(64):
            if j >= out_height*out_width or ((i+j) % 64) >= 49:
                continue
            idx = 128 * (i * 64 + j)
            im2col_packed[(idx % SLOTS) + (idx//SLOTS) * 8] = im2col[j,(i+j) % 64]
    
    return im2col_packed


def get_bsgs_vals(name_base,Input,babysteps,giantsteps,scale,level):
    ret = []
    for (g,gs) in enumerate(giantsteps):
        for (b,bs) in enumerate(babysteps):
            ret.append(Input[f"{name_base}_{bs}_{gs}"][0])
    return ret

def conv2d_io(imageMeta,kernel_weights,kernel_bias):
    Inputs = {}
    OutScales = {}
    out_channels,  kernel_height, kernel_width = kernel_weights.shape
    weights_packed = np.zeros((out_channels,64,SLOTS))
    weights_packed_new = np.zeros((out_channels,64,SLOTS))
    kernel_bias_packed = np.zeros((out_channels,SLOTS))

    sum = np.zeros((out_channels,SLOTS))
    for o in range(out_channels):   
        w64 = np.concat((kernel_weights[o].reshape(-1,1).flatten(), np.zeros(64-49)))
        for i in range(64):
            weights_packed[o,i,0:8192][0::128] = rotate(w64,i)
        kernel_bias_packed[o,0:8192][0::128] = kernel_bias[o]

    for gs in range(4):
        for bs in range(16):
            old_i = bs*4 + gs
            new_i = gs*16 + bs
            weights_packed_new[:,new_i] = weights_packed[:,old_i]
    weights_packed = weights_packed_new


    level = imageMeta.level
    for o in range(out_channels):   
        scale = Primes[level-1]*Primes[level-2]
        inp = bsgs_inputs(f"conv_weight_{o}",weights_packed[o],scale, [bs * 8 for bs in range(16)],[gs * 8192 for gs in range(4)])
        Inputs.update(inp)
        matMeta = ValueMeta(scale,level)
        prodMeta = imageMeta * matMeta
        prodMeta = prodMeta.rescale()
        Inputs[f"conv_bias_{o}"] = (kernel_bias_packed[o],prodMeta.scale)
    prodMeta = prodMeta.rescale().rescale()
    return (Inputs,OutScales,prodMeta)
    
    

def baby_step_giant_step_matmul(v,M_diag,babysteps,giantsteps):
    prod = np.zeros(SLOTS)
    rot_bs = [rotate(v,bs) for bs in babysteps]
    for (g,gs) in enumerate(giantsteps):
        sum_bs = np.zeros(SLOTS)
        for (b,bs) in enumerate(babysteps):
            i = g * len(babysteps) + b
            sum_bs += rotate(M_diag[i],-gs) * rot_bs[b]
        prod += rotate(sum_bs,gs)
    return prod

def bsgs_inputs(basename,M_diag,scale,babysteps,giantsteps):
    Inputs = {}
    for (g,gs) in enumerate(giantsteps):
        for (b,bs) in enumerate(babysteps):
            i = g * len(babysteps) + b
            rot = rotate(M_diag[i],-gs)
            Inputs[f"{basename}_{bs}_{gs}"] = (rot,scale)
    return Inputs


def matmul_256x64_256x1_io(v,M,b):
    Inputs = {}
    OutScales = {}
    M_diag = np.zeros((64,SLOTS))
    for i in range(64):
        for j in range(256):
            idx = i*256 + j
            idx *= 128
            M_diag[idx//SLOTS, idx % SLOTS] = M[(i+j) % 256, j % 64]

    level = v.level
    scale = Primes[level-1]*Primes[level-2]
    inp = bsgs_inputs(f"fc1_w",M_diag,scale, [128*bs for bs in range(8)],[1024 * gs for gs in range(8)])
    Inputs.update(inp)
    matMeta = ValueMeta(scale,level)
    prodMeta = v * matMeta
    prodMeta = prodMeta.rescale()
    prodMeta = prodMeta.rescale()
    
    b_packed = np.zeros(SLOTS)
    b_packed[0::128] = np.tile(b,(4,1)).flatten()
    Inputs[f"fc1_b"] = (b_packed,prodMeta.scale)
    return (Inputs,OutScales,prodMeta)

def matmul_64x10_64x1_io(v,M,b):
    Inputs = {}
    OutScales = {}
    M_diag = np.zeros((16,SLOTS))
    for i in range(16):
        for j in range(64):
            idx = i*64 + j
            idx *= 128
            if j % 16 >= 10:
                continue
            M_diag[idx//8192,idx%8192] = M[(i + j) % 64, j % 16]

    level = v.level
    scale = Primes[level-1]*Primes[level-2]
    inp = bsgs_inputs(f"fc2_w",M_diag,scale, [128*bs for bs in range(4)],[512 * gs for gs in range(4)])
    Inputs.update(inp)
    matMeta = ValueMeta(scale,level)
    prodMeta = v * matMeta
    prodMeta = prodMeta.rescale()
    prodMeta = prodMeta.rescale()
    
    b_packed = np.zeros(SLOTS)
    b_packed[0::128] = np.tile(np.pad(b,pad_width=(0,6)),(16,1)).flatten()
    Inputs[f"fc2_b"] = (b_packed,prodMeta.scale)
    return (Inputs,OutScales,prodMeta)

def get_mnist_program_io(input_image, level):

    with open('parameters/ConvMNIST-0.1.pickle', 'rb') as file:
        parameters = pickle.load(file)
    cw = np.array(parameters["conv1_weight"])
    cb = np.array(parameters["conv1_bias"])
    f1w = np.array(parameters["fc1_weight"])
    f1b = np.array(parameters["fc1_bias"])
    f2w = np.array(parameters["fc2_weight"])
    f2b = np.array(parameters["fc2_bias"])
    Inputs = {}
    ModelInputs = {}
    ModelOutScales = {}
    image_packed = pack_image(input_image, cw, stride=3)
    Scale = 1 << (24*3)
    imageMeta = ValueMeta(Scale,level)
    Inputs["image"] = (image_packed, Scale)


    (convI,convO,convMeta)= conv2d_io(imageMeta , cw, cb)
    ModelInputs.update(convI)
    ModelOutScales.update(convO)
    ModelOutScales["conv"] = convMeta.scale
    conv_sqMeta = convMeta * convMeta
    ModelOutScales["conv_sq"] = conv_sqMeta.rescale().rescale().scale

    (o2I,o2O,o2Meta) = matmul_256x64_256x1_io(conv_sqMeta.rescale(),f1w,f1b)
    ModelInputs.update(o2I)
    ModelOutScales.update(o2O)
    ModelOutScales["o2"] = o2Meta.scale
    o2_sqMeta = o2Meta * o2Meta
    ModelOutScales["o2_sq"] = o2_sqMeta.rescale().rescale().scale

    (o3I,o3O,o3Meta) = matmul_64x10_64x1_io(o2_sqMeta.rescale().rescale(),f2w,f2b)
    ModelInputs.update(o3I)
    ModelOutScales.update(o3O)
    ModelOutScales["pred"] = o3Meta.scale


    Inputs.update(ModelInputs)
    return Inputs, ModelOutScales