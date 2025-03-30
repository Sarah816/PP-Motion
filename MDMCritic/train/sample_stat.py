import torch

motions=torch.load("save/finetuned/mdmft_physcritic_test1/step0/step0-evalbatch.pth")
# print(motions['motion'].shape) # [120, 25, 6, 60]
print('mean score:', sum(motions['score']) / len(motions['score']))