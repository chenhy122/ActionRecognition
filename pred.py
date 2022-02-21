import os
import argparse
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
from dataset import loadedDataset
from model import LSTMModel



parser = argparse.ArgumentParser(description = 'Preding')
parser.add_argument('--model1', default='./save_model/2stream_rgb/', type=str, help = 'path to model1')
parser.add_argument('--model2', default='./save_model/2stream_cmu/', type=str, help = 'path to model2')
parser.add_argument('--data1', default='./dataset/valid_rgb/', type=str, help = 'path of dataset')
parser.add_argument('--data2', default='./dataset/valid_cmu/', type=str, help = 'path of dataset')
parser.add_argument('--twostream', default=True, type=bool, help = 'use 2 stream or not')
parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size')
parser.add_argument('--workers', default=1, type=int, help='number of data loading workers')

args = parser.parse_args()

def pred(pred_loader, model):
	model_pred = {}
	model.eval()
	for _,(inputs, _, filename) in enumerate(pred_loader):
		input_var = [input.cuda() for input in inputs]
		output = model(input_var)
		output = output[:, -1, :]

		pred = output.data.cpu().numpy()
		model_pred[tuple(filename)] = pred
	return model_pred
        

if __name__ == '__main__':
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
	transform = transforms.Compose([transforms.Resize(224),
									transforms.CenterCrop(224),
									transforms.ToTensor(),   ##totensor转为（0,1），仍然需要标准化
									normalize])
 
	dataset = loadedDataset(args.data1, transform)
 
	pred_loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)
  
	##导入已存在的模型
	model_info = torch.load(os.path.join(args.model1, 'model_best.pth.tar'))
	print("==> Loading existing model 1 ")
	original_model = models.__dict__[model_info['arch']](pretrained=False)
	model = LSTMModel(original_model, model_info['arch'],
		model_info['num_classes'], model_info['lstm_layers'], model_info['hidden_size'], model_info['fc_size'])
	# print(model)
	model = model.cuda()
	model.load_state_dict(model_info['state_dict'])
	print("==> Start Prediction 1")
	model_pred = pred(pred_loader, model)
	
	if args.twostream:
		dataset = loadedDataset(args.data2, transform)
		pred_loader = torch.utils.data.DataLoader(
			dataset,
			batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True)
  
		##导入已存在的模型
		model2_info = torch.load(os.path.join(args.model2, 'model_best.pth.tar'))
		print("==> Loading existing model 2 ")
		original_model = models.__dict__[model2_info['arch']](pretrained=False)
		model2 = LSTMModel(original_model, model2_info['arch'],
			model2_info['num_classes'], model2_info['lstm_layers'], model2_info['hidden_size'], model2_info['fc_size'])
		# print(model)
		model2 = model2.cuda()
		model2.load_state_dict(model2_info['state_dict'])
		print("==> Start Prediction 2")
		model_pred2 = pred(pred_loader, model)
  
	total = 0
	for key in model_pred.keys():
		total += len(key)
  
	final_pred = np.zeros((total,50))

	i = 0
	
	print("==> Choose the topk 1")
	for name in sorted(model_pred.keys()):
		pred1 = model_pred[name]
		pred2 = np.zeros_like(pred1)
		if args.twostream:
			pred2 = model_pred2[name]
		final_pred[i: i+pred1.shape[0], :] = (pred1 + pred2)
		i += pred1.shape[0]

	final_pred = torch.from_numpy(final_pred).float()
	print('pred:',final_pred)
	_, result = final_pred.topk(1, 1, True, True)
	print('result',result)
	index = sorted(os.listdir('./dataset/train/'))
	with open('pred_result.txt','w') as f:
		for i in range(len(result)):
			f.write(index[int(result[i])] + '\n')