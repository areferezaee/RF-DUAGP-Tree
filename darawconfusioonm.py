import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import torch
allpreds1 = torch.load('allpreds1.pt', map_location=torch.device('cpu'))
allpreds2 = torch.load('allpreds2.pt', map_location=torch.device('cpu'))
allpreds3 = torch.load('allpreds3.pt', map_location=torch.device('cpu'))
allpreds4 = torch.load('allpreds4.pt', map_location=torch.device('cpu'))
allpreds5 = torch.load('allpreds5.pt', map_location=torch.device('cpu'))
allpreds6 = torch.load('allpreds6.pt', map_location=torch.device('cpu'))
allpreds7 = torch.load('allpreds7.pt', map_location=torch.device('cpu'))
allpreds8 = torch.load('allpreds8.pt', map_location=torch.device('cpu'))
allclses1 = torch.load('allclses1.pt', map_location=torch.device('cpu'))
allclses2 = torch.load('allclses2.pt', map_location=torch.device('cpu'))
allclses3 = torch.load('allclses3.pt', map_location=torch.device('cpu'))
allclses4 = torch.load('allclses4.pt', map_location=torch.device('cpu'))
allclses5 = torch.load('allclses5.pt', map_location=torch.device('cpu'))
allclses6 = torch.load('allclses6.pt', map_location=torch.device('cpu'))
allclses7 = torch.load('allclses7.pt', map_location=torch.device('cpu'))
allclses8 = torch.load('allclses8.pt', map_location=torch.device('cpu'))





allpredss = np.concatenate((allpreds1[0].numpy(), allpreds2[0].numpy(), allpreds3[0].numpy(), allpreds4[0].numpy(), allpreds5[0].numpy(), allpreds6[0].numpy(), allpreds7[0].numpy(), allpreds8[0].numpy()), axis=0)
allclsess = np.concatenate((allclses1[0].numpy(), allclses2[0].numpy(), allclses3[0].numpy(), allclses4[0].numpy(), allclses5[0].numpy(), allclses6[0].numpy(), allclses7[0].numpy(), allclses8[0].numpy()), axis=0)

print(allpredss)
print(allclsess)
classes = ('Intermediate','Expert', 'Novice')
cf_matrix = confusion_matrix(allclsess, allpredss)

row_percent = cf_matrix.astype(float)
row_percent = row_percent / row_percent.sum(axis=1)[:, np.newaxis] * 100

df_cm = pd.DataFrame(
    row_percent,
    index=classes,
    columns=classes
)

plt.figure(figsize=(12,7))
sn.heatmap(df_cm, annot=True, fmt=".1f")
plt.title("Normalized Confusion Matrix (%)")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.savefig("outputSuturing.png")

