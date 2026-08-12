
import torch.nn.functional as F
import torch

train = 'data/JIGSAWS/Splits/Knot_Tying/meta_file_Knot_Tying_training.txt'
val = 'data/JIGSAWS/Splits/Knot_Tying/meta_file_Knot_Tying_val.txt'
test = 'data/JIGSAWS/Splits/Knot_Tying/meta_file_Knot_Tying_test.txt'
root2 = 'data/JIGSAWS/Splits/Knot_Tying/clip-level-flow-feats/'
root = 'data/JIGSAWS/Splits/Knot_Tying/clip-level-feats/'
batch_size = 1

device = torch.device('cpu')#('cuda')

from feature_engineering.jigsaws_loader import JL
dataset_tr = JL(train, root2)
dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
dataset = JL(val, root2)
vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

dataset_ts = JL(test, root2)
tst = torch.utils.data.DataLoader(dataset_ts, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
dataloader = {'train':dl, 'val':vdl, 'test':tst}
def cat(u, t, dim):
      if dim==1:
         return torch.cat((u,t), dim).unsqueeze(0)
      elif dim==0:
           return torch.cat((u,t), dim)

def get_entropy1(feats, eps):
    enerG = feats**2
    P = torch.clamp(enerG/(enerG.sum() + eps), min=eps) 
    entropy = torch.abs(torch.std((-P*torch.log(P + eps)), dim=2))
    return entropy

def get_noise(feats, eps):
    enerG = feats**2
    Pj = torch.clamp(enerG/(enerG.sum() + eps), min=eps) 
    noise = (torch.abs(torch.std((-Pj*torch.log(Pj + eps)), dim=1)).sum())
    return torch.tensor(noise).unsqueeze(0).to(device)

def get_entropy2(feats, eps):
    feat_std = torch.std(feats, dim=2)

    enerG = feat_std**2
    P = torch.clamp(enerG/(enerG.sum() + eps), min=eps) 
    entropy = torch.abs((-P*torch.log(P + eps)))
    return entropy

def prepare1(feat):
    return torch.flatten(feat.mean(0).mean(1)).unsqueeze(0)

def prepare2(feat):
    return feat.mean(0).unsqueeze(0)

def get_feats1(feats, eps):
    feat1 = torch.diff(feats, dim =0)
    entropy = prepare2(get_entropy1(feat1, eps))
    if feat1.size(0)<2:
        feat1 = torch.cat((feat1, feat1), dim =0)
    preparedfeat1 = prepare1(feat1)
    return preparedfeat1 , feat1, entropy

def get_feats2(feats, eps):
    feat1 = torch.diff(feats, dim =0)
    entropy = prepare2(get_entropy2(feat1, eps))
    if feat1.size(0)<2:
        feat1 = torch.cat((feat1, feat1), dim =0)
    preparedfeat1 = prepare1(feat1)
    return preparedfeat1 , feat1, entropy

def get_jigsaws_feats(pahse):
    k = 0 
    eps= 1e-8
    
    for vid_feats1, cls, index, name in dataloader[pahse]:
            vid_feats2 = torch.load(root + name[0], map_location=torch.device('cpu'))
            vid_feats1 = vid_feats1[0].to(device)
            
            
            entrotpyf1 = prepare2(get_entropy1(vid_feats1, eps))
            entrotpyf2 = prepare2(get_entropy2(vid_feats2, eps))
            v1_feats = prepare1(vid_feats1)
            v2_feats = prepare1(vid_feats2)
            velocity1, velo1, entropyv1 = get_feats1(vid_feats1, eps)
            acceleration1, acceler1, entropya1 = get_feats1(velo1, eps)
            jerk1, j1, entropyj1 = get_feats1(acceler1, eps)
            noise = get_noise(j1, eps)

            velocity2, velo2, entropyv2 = get_feats2(vid_feats2, eps)
            acceleration2, acceler2, entropya2 = get_feats2(velo2, eps)
            jerk2, j2, entropyj2 = get_feats2(acceler2, eps)

            f = v1_feats
            f = cat(f, v2_feats, 1)
            f = cat(f, cat(velocity1, velocity2, 1), 0)
            f = cat(f, cat(acceleration1, acceleration2, 1), 0)
            f = cat(f, cat(jerk1, jerk2, 1), 0)
            f = cat(f, cat(entrotpyf1, entrotpyf2, 1), 0)
            f = cat(f, cat(entropyv1, entropyv2, 1), 0)
            f = cat(f, cat(entropya1, entropya2, 1), 0)
            f = cat(f, cat(entropyj1, entropyj2, 1), 0)
            
            f = torch.flatten(f).unsqueeze(0)
            
            cls = cls.to(device)
            
            index = index.to(device)

            X = torch.cat((X, f), dim=0) if k > 0 else f
            Y = torch.cat((Y, cls), dim=0) if k > 0 else cls
            X_idx = torch.cat((X_idx, index), dim=0) if k > 0 else index
            nis = torch.cat((nis, noise), dim=0) if k > 0 else noise
            k += 1
            
            
    print(X.size(), Y.size(), X_idx.size())        
    return X, Y, X_idx, nis
