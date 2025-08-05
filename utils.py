import numpy as np
import os
from PIL import Image
# from synets_to_names import synets_to_labels
from torch.utils.data import Dataset

def compute_ap(ranks, nres):
    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=[]):
    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    recall_sum = np.zeros(len(kappas))
    recalls = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            recalls[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
            recalls[i, j] = (pos <= kappas[j]).sum() / float(len(qgnd))
        pr = pr + prs[i, :]
        recall_sum = recall_sum + recalls[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)
    mean_recalls = recall_sum / (nq - nempty)

    return map, aps, pr, prs, mean_recalls, recalls

def prepare_struct(data_root):

    imagenet_classes = {}
    for keys in synets_to_labels.keys():
        imagenet_classes[synets_to_labels[keys]['id']] = synets_to_labels[keys]['label'].split(',')[0]
    folderlist = sorted(os.listdir(data_root))
    filelist = []
    for i in folderlist:
        files = os.listdir(os.path.join(data_root, i))
        for file in files:
            filename = os.path.join(i,file)
            filelist.append(filename)
    gnd = {}

    for i, path in enumerate(filelist):
        folder = path.split("/")[0]  # Extract the folder name
        idx = folderlist.index(folder)
        if idx in gnd:
            gnd[idx]['ok'].append(i)  # Add the index to the corresponding folder's list
        else:
            gnd[idx]={}
            gnd[idx]['ok']=[]
            gnd[idx]['ok'].append(i)  # Add the index to the corresponding folder's list
    return gnd, filelist


# for vladan
class ImageNetQuerysetOrder(Dataset):
    def __init__(self, root_dir, transform=None, q_idx=0, preprocess=None, test=None):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.filelist = []
        self.q_idx = q_idx
        folderlist = sorted(os.listdir(root_dir))
        # folderlist = test
        for i in folderlist:
            files = os.listdir(os.path.join(root_dir, i))[0]
            name = files.rsplit('_',1)[0] +"_"+str(q_idx)+".png"
            filename = os.path.join(i, name)
            # filename = os.path.join(i, str(i)+"_"+str(q_idx)+".png")
            self.filelist.append(filename)
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.filelist[idx])
        # print(img_name)
        image = self.preprocess(Image.open(img_name)).unsqueeze(0)
        return idx, image

# old one
class ImageNetQueryset(Dataset):
    def __init__(self, root_dir, transform=None, q_idx=0, preprocess=None, test=None):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.filelist = []
        self.q_idx = q_idx
        folderlist = sorted(os.listdir(root_dir))
        # folderlist = test
        for i in folderlist:
            files = os.listdir(os.path.join(root_dir, i))
            filename = os.path.join(i, files[self.q_idx])
            self.filelist.append(filename)
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.filelist[idx])
        # print(img_name)
        image = self.preprocess(Image.open(img_name)).unsqueeze(0)
        return idx, image

def prepare_generic_struct(data_root):

    folderlist = sorted(os.listdir(data_root))
    filelist = []
    for i in folderlist:
        files = sorted(os.listdir(os.path.join(data_root, i)))
        for file in files:
            filename = os.path.join(i,file)
            filelist.append(filename)
    gnd = {}

    for i, path in enumerate(filelist):
        folder = path.split("/")[0]  # Extract the folder name
        idx = folderlist.index(folder)
        if idx in gnd:
            gnd[idx]['ok'].append(i)  # Add the index to the corresponding folder's list
        else:
            gnd[idx]={}
            gnd[idx]['ok']=[]
            gnd[idx]['ok'].append(i)  # Add the index to the corresponding folder's list
    return gnd, filelist


results = {}
results['caltech'] = {}
results['caltech']['text'] = 0.908
results['caltech']['image'] = 0.886
results['caltech']['joint'] = 0.917
results['caltech']['0.1'] = 0.934

results['cars'] = {}
results['cars']['text'] = 0.643
results['cars']['image'] = 0.224
results['cars']['joint'] = 0.393
results['cars']['0.1'] = 0.649

results['cifar10'] = {}
results['cifar10']['text'] = 0.884
results['cifar10']['image'] = 0.773
results['cifar10']['joint'] = 0.888
results['cifar10']['0.1'] = 0.927

results['cifar100'] = {}
results['cifar100']['text'] = 0.616
results['cifar100']['image'] = 0.783
results['cifar100']['joint'] = 0.825
results['cifar100']['0.1'] = 0.778

results['dtd'] = {}
results['dtd']['text'] = 0.419
results['dtd']['image'] = 0.371
results['dtd']['joint'] = 0.460
results['dtd']['0.1'] = 0.480

results['fgvc'] = {}
results['fgvc']['text'] = 0.283
results['fgvc']['image'] = 0.090
results['fgvc']['joint'] = 0.170
results['fgvc']['0.1'] = 0.289

results['flowers'] = {}
results['flowers']['text'] = 0.764
results['flowers']['image'] = 0.645
results['flowers']['joint'] = 0.746
results['flowers']['0.1'] = 0.791

results['food'] = {}
results['food']['text'] = 0.883
results['food']['image'] = 0.740
results['food']['joint'] = 0.818
results['food']['0.1'] = 0.908

results['imagenet'] = {}
results['imagenet']['text'] = 0.648
results['imagenet']['image'] = 0.633
results['imagenet']['joint'] = 0.691
results['imagenet']['0.1'] = 0.729

results['k700'] = {}
results['k700']['text'] = 0.362
results['k700']['image'] = 0.287
results['k700']['joint'] = 0.342
results['k700']['0.1'] = 0.408

results['pets'] = {}
results['pets']['text'] = 0.880
results['pets']['image'] = 0.847
results['pets']['joint'] = 0.883
results['pets']['0.1'] = 0.910

results['r45'] = {}
results['r45']['text'] = 0.643
results['r45']['image'] = 0.459
results['r45']['joint'] = 0.587
results['r45']['0.1'] = 0.660

results['sun'] = {}
results['sun']['text'] = 0.543
results['sun']['image'] = 0.527
results['sun']['joint'] = 0.576
results['sun']['0.1'] = 0.623

results['ucf'] = {}
results['ucf']['text'] = 0.662
results['ucf']['image'] = 0.596
results['ucf']['joint'] = 0.659
results['ucf']['0.1'] = 0.723