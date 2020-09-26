import pdb
import cv2
import torch
from torch.autograd import Variable
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from util.label_util import label_to_color_mask

model = torch.load('weights/best.pt').cuda()
transform = Compose([ToPILImage(), Resize((255, 846)), ToTensor()])
video = cv2.VideoWriter("demo.avi",
                        cv2.VideoWriter_fourcc('I', '4', '2', '0'), 25,
                        (846, 255))
with torch.no_grad():
    model.eval()
    cap = cv2.VideoCapture('demo.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            break
        src = img = img[690:, :, :]
        img = transform(img).unsqueeze(0)
        img = Variable(img).cuda()
        label = model(img)
        label = torch.argmax(label, dim=1)
        mask = label_to_color_mask(label.cpu())

        ia = cv2.resize(src, (846, 255))
        m = mask.squeeze(0)
        #import pdb; pdb.set_trace()
        ia = cv2.add(ia, m)
        video.write(ia)
        cv2.imshow('image', ia)
        k = cv2.waitKey(20)
        if (k & 0xff == ord('q')):
            break
    video.release()
    cap.release()
    cv2.destroyAllWindows()
