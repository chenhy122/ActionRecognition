import os
import pickle
import shutil
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import time
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
from dataset import loadedDataset
from model import LSTMModel
from utils import AverageMeter


parser = argparse.ArgumentParser(description = 'Training')
parser.add_argument('--model', default='./save_model/2stream_rgb/', type=str, help = 'path to save model')
parser.add_argument('--data', default='./dataset/train_rgb/', type=str, help = 'path of dataset')
parser.add_argument('--arch', default = 'resnet50', help = 'model architecture')
parser.add_argument('--lstm_layers', default=2, type=int, help='number of lstm layers')
parser.add_argument('--hidden_size', default=512, type=int, help='output size of LSTM hidden layers')
parser.add_argument('--fc_size', default=1024, type=int, help='size of fully connected layer before LSTM')					
parser.add_argument('--epochs', default=200, type=int, help='manual epoch number')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr_step', default=100, type=float, help='learning rate decay frequency')
parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay rate")
parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')

args = parser.parse_args()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, os.path.join(args.model, filename))
	##保存最佳参数设置
	if is_best:
		shutil.copyfile(os.path.join(args.model, filename), os.path.join(args.model, 'model_best.pth.tar'))

##每100个epoch，lr减少10%
def adjust_learning_rate(optimizer, epoch):
	if not epoch % args.lr_step and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
	return optimizer


def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def train(train_loader, model, criterion, optimizer, epoch):
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	model.train()

	for i, (inputs, target, _) in enumerate(train_loader):
		input_var = [input.cuda() for input in inputs]
		target_var = target.cuda()

		##计算loss
		output = model(input_var)
		output = output[:, -1, :]
		loss = criterion(output, target_var)
		losses.update(loss.item(), 1)

		##计算准确率
		prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 5))
		top1.update(prec1[0].item(), 1)
		top5.update(prec5[0].item(), 1)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print('Epoch: [{0}][{1}/{2}]\t'
			'lr {lr:.5f}\t'
			'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
			'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
			'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
			epoch, i, len(train_loader),
			lr=optimizer.param_groups[-1]['lr'],
			loss=losses,
			top1=top1,
			top5=top5))
	return (top1.avg, top5.avg, losses.avg)

def validate(val_loader, model, criterion):
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	dic_video_pred = {}	# 保存预测结果的字典
	lab_target = {}		# 保存预测目label的字典

	model.eval()

	for i, (inputs, target, filename) in enumerate(val_loader):
		input_var = [input.cuda() for input in inputs]
		target_var = target.cuda()

		##计算loss
		with torch.no_grad():
			output = model(input_var)
			output = output[:, -1, :]

			pred = output.data.cpu().numpy()
			dic_video_pred[tuple(filename)] = pred
			lab_target[tuple(filename)] = target


			loss = criterion(output, target_var)
			losses.update(loss.item(), 1)

		##计算准确率
		prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 5))
		top1.update(prec1[0].item(), 1)
		top5.update(prec5[0].item(), 1)

		print ('Val: [{0}/{1}]\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				i, len(val_loader),
				loss=losses,
				top1=top1,
				top5=top5))

	return (top1.avg, top5.avg, losses.avg, dic_video_pred, lab_target)

if __name__ == '__main__':
    
	##从Imagenet数据库计算得到的均值和方差，较为常用
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])

	transform = transforms.Compose([transforms.Resize(224),
									transforms.CenterCrop(224),
									transforms.ToTensor(),   ##totensor转为（0,1），仍然需要标准化
									normalize])

	##读取数据并按照8:2划分训练集和验证集
	dataset = loadedDataset(args.data, transform)
	torch.manual_seed(1234)
	train_data,val_data = torch.utils.data.random_split(dataset,[2800,700])

	train_loader = torch.utils.data.DataLoader(
		train_data,
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	val_loader = torch.utils.data.DataLoader(
		val_data,
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)	

	##检查是否存在已存在的模型
	if os.path.exists(os.path.join(args.model, 'checkpoint.pth.tar')):
		##导入已存在的模型
		model_info = torch.load(os.path.join(args.model, 'checkpoint.pth.tar'))
		print("==> loading existing model '{}' ".format(model_info['arch']))
		original_model = models.__dict__[model_info['arch']](pretrained=False)
		model = LSTMModel(original_model, model_info['arch'],
			model_info['num_classes'], model_info['lstm_layers'], model_info['hidden_size'], model_info['fc_size'])
		# print(model)
		model = model.cuda()
		model.load_state_dict(model_info['state_dict'])
		best_prec = model_info['best_prec']
		cur_epoch = model_info['epoch']
		train_loss = model_info['train_loss']
		val_loss = model_info['val_loss']
		top1_total = model_info['train_top1']
		pred1_total = model_info['val_pred1']
	else:
		if not os.path.isdir(args.model):
			os.makedirs(args.model)
		# load and create model
		print("==> creating model '{}' ".format(args.arch))
		original_model = models.__dict__[args.arch](pretrained=True)
		model = LSTMModel(original_model, args.arch,
			len(dataset.classes), args.lstm_layers, args.hidden_size, args.fc_size)
		#print(model)
		model = model.cuda()
		best_prec = 0
		cur_epoch = 0
		train_loss = []
		val_loss = []
		top1_total = []
		pred1_total = []
	##构建优化器
	criterion = nn.CrossEntropyLoss()

	optimizer = torch.optim.Adam([{'params': model.fc_pre.parameters()},
								{'params': model.rnn.parameters()},
								{'params': model.fc.parameters()}],
								lr=args.lr,
								weight_decay=args.weight_decay)

	start = time.time()
	##训练
	for epoch in range(cur_epoch, args.epochs):
		optimizer = adjust_learning_rate(optimizer, epoch)

		print('---------------------------------------------------Training---------------------------------------------------')

		top1, top5, train_losses = train(train_loader, model, criterion, optimizer, epoch)

		print('------Training Result------')
		print('   Top1 accuracy: {prec: .2f} %'.format(prec=top1))
		print('   Top5 accuracy: {prec: .2f} %'.format(prec=top5))
		print('-----------------------------')
		top1_total.append(top1)
		train_loss.append(train_losses)

		##验证
		print("--------------------------------------------------Validation--------------------------------------------------")

		prec1, prec5, val_losses, dic_video_pred, lab_target = validate(val_loader, model, criterion)

		print("------Validation Result------")
		print("   Top1 accuracy: {prec: .2f} %".format(prec=prec1))
		print("   Top5 accuracy: {prec: .2f} %".format(prec=prec5))
		print("-----------------------------")
		pred1_total.append(prec1)
		val_loss.append(val_losses)

		##保存checkpoint
		is_best = prec1 > best_prec
		best_prec = max(prec1, best_prec)
		if is_best:
			with open('record/rgb_pred.pickle', 'wb') as f:	# 保存为pickle文件
				pickle.dump(dic_video_pred, f)
			f.close()
			with open('record/lab_target.pickle', 'wb') as f:
				pickle.dump(lab_target, f)
			f.close()
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'num_classes': len(dataset.classes),
			'lstm_layers': args.lstm_layers,
			'hidden_size': args.hidden_size,
			'fc_size': args.fc_size,
			'state_dict': model.state_dict(),
			'best_prec': best_prec,
			'train_loss':train_loss,
			'train_top1':top1_total,
			'val_loss':val_loss,
			'val_pred1':pred1_total,
			'optimizer' : optimizer.state_dict(),}, is_best)
	
	Etime = time.time() - start
	print('Elapsed time: ',time.strftime("%H:%M:%S", time.gmtime(Etime)))

	##以txt格式输出部分训练结果
	model_info = torch.load(os.path.join(args.model, 'model_best.pth.tar'))
	with open(os.path.join(args.model,'result.txt'),'w') as f:
		f.write(f'''best epoch: {model_info['epoch']} \narch: {model_info['arch']} \nnum_classes: {model_info['num_classes']} \nlstm_layers: {model_info['lstm_layers']} 
hidden_size: {model_info['hidden_size']} \nfc_size: {model_info['fc_size']} \nbest_prec: {model_info['best_prec']} \nElapsed time: {time.strftime("%H:%M:%S", time.gmtime
	(Etime))}''')

	##绘制并保存loss图
	plt.figure()
	plt.plot(train_loss,label='Train Loss',c='r')
	plt.plot(val_loss,label='Validation Loss',c='b')
	plt.legend()
	plt.xlabel('Epochs',fontsize=15)
	plt.ylabel('Loss',fontsize=15)
	plt.savefig(args.model+'loss_convergence_epochs.png')

	##绘制并保存Top1准确率图
	plt.figure()
	plt.plot(top1_total,label='Train Accuracy',c='r')
	plt.plot(pred1_total,label='Validation Accuracy',c='b')
	plt.legend()
	plt.xlabel('Epochs',fontsize=15)
	plt.ylabel('Accuracy',fontsize=15)
	plt.savefig(args.model+'accuracy_epochs.png')