import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TORCH_HOME']='/root/autodl-tmp/torch_home'
os.environ['HF_HOME']='/root/autodl-tmp/huggingface_cache'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'

import torch
from opts import parser
from torchvision.transforms import Compose, InterpolationMode, Normalize, ToTensor
import torch.distributed as dist
from PIL import Image
import torchvision.transforms as T
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
args = parser.parse_args()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)

from torch.utils.tensorboard import SummaryWriter
if local_rank == 0:
    tb_logger = SummaryWriter(log_dir=os.path.join(args.exp_dir,'board'),flush_secs=10)
checkpoint_dir = os.path.join(args.exp_dir,'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
log_training = open(os.path.join(args.exp_dir,'log.csv'), 'w')
log_training.write(str(args))

device =   "cuda"
from st_adapter_DyKnow import clip_vit_base_patch16_adapter24x384,clip_vit_base_patch16

# from ema import EMA
import torchvision.transforms as T

model =clip_vit_base_patch16_adapter24x384(num_classes=1,args = args,lora_rank =args.lora_rank ).to(device).train()
val_transform =T.Compose(
[
T.Resize(size=256, antialias=True),
ToTensor(),
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
T.CenterCrop((224,224))
]
)  


if os.path.isfile(args.resume_main):
    print(("=> loading checkpoint '{}'".format(args.resume_main)))
    checkpoint = torch.load(args.resume_main,map_location='cpu')
    checkpoint_model  = checkpoint["model"]
    ckeck_copy = {}
    ks  =checkpoint_model.keys()
    for k in ks:
        new_k = k
        if k.startswith("module"):
            if "router_MLP.2" in k or "knowledge_pool" in k:
                if args.lora_rank<0:
                    continue
            new_k = k[7:]
        ckeck_copy[new_k] = checkpoint_model[k]
    model.load_state_dict(ckeck_copy,strict=False)
    


contain_trainable = False
for name, param in model.named_parameters():
    if param.requires_grad:
        contain_trainable = True
        break
if contain_trainable:
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()],\
                broadcast_buffers=False , find_unused_parameters=True)
else:
    model = model.cuda()


## we accept both single-image (such as Xray, fundus) and sequence (such as CT) medical images.
## The input dimension is (B,C,H,W) or (B,L,H,W), where L is the sequence length

## 
img_path = "test_img.jpg"
img = Image.open(img_path).convert('RGB')
img_tensor = val_transform(img)
image_tensor_example = img_tensor.unsqueeze(0)
input = image_tensor_example
if len(input.size()) ==4:
    input = input.unsqueeze(2)
ft_x,_ = model(input.to(device))

print("ID tensor", ft_x)