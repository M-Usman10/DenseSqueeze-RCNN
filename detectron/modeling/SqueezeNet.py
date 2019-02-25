from caffe2.python import brew
sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

class SqueezeNet:
  def __init__(self,model,Data="data"):
    self.data_format=1
    self.model=model
    self.prefix = ""
    self.output=self.create_model(Data)

  def fire_module(self,x,inp,fire_id,sq,exp_):
    s_id = 'fire' + str(fire_id) + '/'
    x=brew.conv(self.model, x, s_id + sq1x1,inp, sq, 1)
    x=brew.relu(self.model,x,s_id + relu + sq1x1)
    left=brew.conv(self.model, x, s_id + exp1x1,sq, exp_, 1)
    left=brew.relu(self.model,left,s_id + relu + exp1x1)
    right=brew.conv(self.model, x, s_id + exp3x3,sq, exp_, 3,legacy_pad=2)
    right=brew.relu(self.model,right,s_id + relu + exp3x3)
    return brew.concat(self.model,[left,right],self.prefix+s_id + 'concat',axis=self.data_format),exp_*2
  def create_model(self,x):
    x = brew.conv(self.model, x, self.prefix+'conv1',3, 64, 3,stride=2)
    x = brew.relu(self.model, x,self.prefix+'relu_conv1')
    x = brew.max_pool(self.model, x, self.prefix+'pool1', kernel=3, stride=2,legacy_pad=2)
    x,inp = self.fire_module(x, fire_id=2, sq=16, exp_=64,inp=64)
    x,inp = self.fire_module(x, fire_id=3, sq=16, exp_=64,inp=inp)
    x = brew.max_pool(self.model, x, self.prefix+'pool3', kernel=3, stride=2,legacy_pad=2)
    x,inp = self.fire_module(x, fire_id=4, sq=32, exp_=128,inp=inp)
    x,inp = self.fire_module(x, fire_id=5, sq=32, exp_=128,inp=inp)
    x = brew.max_pool(self.model, x, self.prefix+'pool5', kernel=3, stride=2,legacy_pad=2)
    x,inp = self.fire_module(x, fire_id=6, sq=48, exp_=192,inp=inp)
    x,inp = self.fire_module(x, fire_id=7, sq=48, exp_=192,inp=inp)
    x,inp = self.fire_module(x, fire_id=8, sq=64, exp_=256,inp=inp)
    x,inp = self.fire_module(x, fire_id=9, sq=64, exp_=256,inp=inp)

    return x

def create_squeezeNet(model):
  s=SqueezeNet(model,"data");
  return s.output, 512, 1. / 16.