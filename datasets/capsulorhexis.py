
import torch.nn.functional as F
import torch

train = 'data/5_Skill_Assessment/annot2.txt'
val = 'data/5_Skill_Assessment/annotv.txt'
test = 'data/5_Skill_Assessment/annott.txt'
root = 'data/5_Skill_Assessment/clip-level-feats/'
root2 = 'data/5_Skill_Assessment/clip-level-flowfeats/'
batch_size = 1

device = torch.device('cpu')#('cuda')

from datasets.capsulorhexis_loader import CL
dataset_tr = CL(train, root2)
dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
dataset = CL(val, root2)
vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

dataset_ts = CL(test, root2)
tst = torch.utils.data.DataLoader(dataset_ts, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
dataloader = {'train':dl, 'val':vdl, 'test':tst}



def compute_gram_clips(x):
   x = x/(x.norm(dim=-1, keepdim=True) + 1e-6)
   #gram = torch.matmul(x, x.transpose(1,2))  
   gram = torch.bmm(x, x.transpose(1,2))
   return gram

def upper_tri_features(G):

    idx = torch.triu_indices(G.size(0), G.size(1), offset=1)
    feat = G[idx[0], idx[1]]
    return feat

def row_stats(G):
    mean_sim = G.mean(dim=-1).mean(-1).mean(-1).unsqueeze(0)
    std_sim = G.std(dim=-1).std(-1).std(-1).unsqueeze(0)
    max_sim, idx = torch.max(G)
    feat = torch.cat([mean_sim, std_sim], dim=0)
    return feat

def eigen_features(G, k=10):

    eigvals = torch.linalg.eigvalsh(G)
    feat = eigvals[-k:]
    return feat

def energy_features(G):
    trace = torch.trace(G)
    fro = torch.norm(G, p='fro')
    return torch.tensor([trace, fro])

def histogram_feature(G, bins=32):
    hist = torch.hist(G, bins=bins, min=1.0, max=1.0)
    return hist
def get_entropy(feats):
    eps = 1e-8
    enerG = feats**2
    Pf = torch.clamp(enerG/(enerG.sum() + eps), min=eps) 
    entropy = (-(Pf*torch.log(Pf + eps)).sum()).unsqueeze(0)
    #entropy = torch.abs(torch.std((-(Pf*torch.log(Pf + eps)))))
    return entropy, enerG.mean(-1).mean(-1).mean(-1).unsqueeze(0).unsqueeze(0)

def get_mu_std(feats):
    mu = feats.mean(-1).mean(-1).mean(-1).unsqueeze(0).unsqueeze(0)
    std = torch.std((torch.std((torch.std(feats, dim=-1)), dim=-1)), dim=-1, correction=0).unsqueeze(0).unsqueeze(0)
    return mu, std

def cat (u,t, dim):
    return torch.cat((u,t), dim)

def get_f_features(feats1, feats2):

    num_clips = torch.tensor(feats1.size(0), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    f_embedding1 = feats1.mean(2)
    f_embedding2 = feats2.mean(2)
    entropyf1, _ = get_entropy(f_embedding1)
    entropyf2, _ = get_entropy(f_embedding2)
    mu1, std1 = get_mu_std(feats1)
    mu2, std2 = get_mu_std(feats2)
    
    f1 = cat(mu1, std1, 1)
    f2 = cat(mu2, std2, 1)
    #f = cat(f1, f2, 1)
    
    #entropyf = cat(entropyf1.unsqueeze(0), entropyf2.unsqueeze(0), 1)
    entropyf = entropyf1
    f = cat(f1, num_clips, 1)#cat(f, num_clips, 1)
    return f, entropyf

def get_features(feats1, feats2):
    
    delta1 = torch.diff(feats1, dim =0)
    delta2 = torch.diff(feats2, dim =0)
    nis_1, enerG1 = get_entropy(delta1.mean(1))
    nis_2, enerG2 = get_entropy(delta2.mean(1))
    mu1, std1 = get_mu_std(delta1)
    mu2, std2 = get_mu_std(delta2)
    fe1 = cat(mu1, std1, 1)
    fe2 = cat(mu2, std2, 1)
    fe1 = cat(fe1, enerG1, 1)
    fe2 = cat(fe2, enerG2, 1)
    #fe = cat(fe1, fe2, 1)
    #nis = cat(nis_1.unsqueeze(0), nis_2.unsqueeze(0),1)
    nis = nis_1
    fe = fe1
    return fe, nis, delta1, delta2

def get_j_features(feats1, feats2):
    delta1 = torch.diff(feats1, dim =0)
    delta2 = torch.diff(feats2, dim =0)
    nis_1, enerG1 = get_entropy(delta1.mean(1))
    nis_2, enerG2 = get_entropy(delta2.mean(1))
    mu1, std1 = get_mu_std(delta1)
    mu2, std2 = get_mu_std(delta2)
    
    fe1 = cat(std1, enerG1, 1)
    #fe2 = cat(std2, enerG2,1)
    #fe = cat(fe1, fe2, 1)
    #nis = cat(nis_1.unsqueeze(0), nis_2.unsqueeze(0), 1)
    nis = nis_1
    fe = fe1
    return fe, nis,  delta1, delta2
    

def get_capsulorhexis_feats(pahse):
    k = 0 
    X = []
    eps= 1e-8
    
    for vid_feats1, cls, index, name in dataloader[pahse]:
            
            vid_feats2 = torch.load(root + name[0], map_location=torch.device('cpu'))   
            vid_feats1 = vid_feats1[0,0].to(device)   
            
            f, entropy_f = get_f_features(vid_feats1, vid_feats2)
            v, nis_v, deltav1, deltav2 = get_features(vid_feats1, vid_feats2)
            nis = nis_v
           
            if deltav1.size(0) < 2:
               deltav1 = cat(deltav1, deltav1, 0)
               deltav2 = cat(deltav2, deltav2,0)

            a, nis_a, deltaa1, deltaa2 = get_features(deltav1, deltav2)
            nis += nis_a
            if deltaa1.size(0) < 2:
               deltaa1 = cat(deltaa1, deltaa1, 0)
               deltaa2 = cat(deltaa2, deltaa2, 0)

            j, nis_j, deltaj1, deltaj2 = get_j_features(deltaa1, deltaa2)
            nis += nis_j
            if deltaj1.size(0) < 2:
               deltaj1 = cat(deltaj1, deltaj1, 0)
               deltaj2 = cat(deltaj2, deltaj2, 0)

            va = cat(v, a, 1)
            vaj = cat(va, j, 1)
            f = cat(f, vaj, 1)
            nis_va = cat(nis_v.unsqueeze(0), nis_a.unsqueeze(0), 1)
            nis_vaj = cat(nis_va, nis_j.unsqueeze(0), 1)
            all_nis = cat(entropy_f.unsqueeze(0), nis_vaj, 1)
            f = cat(f, all_nis, 1)


            
            cls = cls.to(device)
            index = index.to(device)
            main_nis = torch.cat((main_nis, nis), dim=0) if k>0 else nis
            X = torch.cat((X, f), dim=0) if k > 0 else f
            Y = torch.cat((Y, cls), dim=0) if k > 0 else cls
            X_idx = torch.cat((X_idx, index), dim=0) if k > 0 else index
            k+=1
            
    return X, Y, X_idx, main_nis
