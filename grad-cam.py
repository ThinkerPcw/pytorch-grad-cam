import torch
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
import pylab as plt
import numpy as np
import cv2


class Extractor():
    """ 
    pytorch在设计时，中间层的梯度完成回传后就释放了
    这里用hook工具在保存中间参数的梯度
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient=grad

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name,module in self.model.features._modules.items():
            x = module(x)
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                target_activation=x
        x=x.view(1,-1)
        for name,module in self.model.classifier._modules.items():
            x = module(x)
        # 维度为（1，c, h, w） , (1,class_num)
        return target_activation, x


def preprocess_image(path):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]
    m_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(means,stds)])
    img=Image.open(path)
    return m_transform(img).reshape(1,3,224,224)


class GradCam():
    def __init__(self, model, target_layer_name, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = Extractor(self.model, target_layer_name)

    
    def __call__(self, input, index = None):
        if self.cuda:
            target_activation, output = self.extractor(input.cuda())
        else:
            target_activation, output = self.extractor(input)

        # index是想要查看的类别，未指定时选择网络做出的预测类
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        # batch维为1（我们默认输入的是单张图）
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1.0
        one_hot = torch.tensor(one_hot)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.gradient.cpu().data.numpy()
        # 维度为（c, h, w）
        target = target_activation.cpu().data.numpy()[0]
        # 维度为（c,）
        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        # cam要与target一样大
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # 每个位置选择c个通道上最大的最为输出
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam2.jpg", np.uint8(255 * cam))


# target_layer 越靠近分类层效果越好
grad_cam = GradCam(model = models.vgg19(pretrained=True), target_layer_name = "35", use_cuda=True)
input = preprocess_image("examples/both.png")
mask = grad_cam(input, None)
img = cv2.imread("examples/both.png", 1)
# 热度图是直接resize加到输入图上的
img = np.float32(cv2.resize(img, (224, 224))) / 255
show_cam_on_image(img, mask)
