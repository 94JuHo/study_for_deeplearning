```python
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
import torch.utils.data as data_utils  
```
```
nn : 딥러닝 모델에 필요한 모듈이 모아져 있는 패키지   
ex) nn.Linear(128, 128), nn.ReLU()

F : nn과 같은 모듈이 모아져 있지만, 함수의 input으로 반드시 연산이 되어야 하는 값을 받음  
ex) F.linear(X, 128, 128), F.ReLU(X)

optim: 학습에 관련된 optimizing method가 있는 패키지  

data_utils : batch generator 등 학습 데이터에 관련된 패키지
```

## 모델 뼈대 코드
```python
class MyModel(nn.Module):
    
    def __init__(self, X_dim, y_dim):
        super(MyModel, self).__init__()
        layer1 = nn.Linear(X_dim, 128)
        activation1 = nn.ReLU()
        layer2 = nn.Linear(128, y_dim)
        self.module = nn.Sequential(
            layer1,
            activation1,
            layer2
        )
        
    def forward(self, x):
        out = self.module(x)
        result = F.softmax(out, dim=1)
        return result
```
```
class 형태의 모델은 항상 nn.Module 을 상속받아야 하며, super(모델명, self).__init__() 을 통해 nn.Module.__init__() 을 실행시키는 코드가 필요합니다.  

forward() : 모델이 학습데이터를 입력받아서 forward propagation을 진행시키는 함수이고, 반드시 forward 라는 이름의 함수이어야 합니다.
```

## 학습 뼈대 코드
```python
# 준비재료
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-5
optimizer = optim.SGD(model.paraeters(), lr=learning_rate)
num_epochs = 2
num_batches = len(train_loader)

for epoch in range(num_epochs):
	for i, data in enumerate(train_loader):
		x, x_labels = data # x.size() = [batch, channel, x, y]
		# init grad
		optimizer.zero_grad() # step과 zero_grad는 쌍을 이루는 것이라고 생각하면 됨
		# forward
		pred = model(x)
		# calculate loss
		loss = criterion(pred, x_labels)
		# backpropagation
		loss.backward()
		# weight update
		optimizer.step()
		# 학습과정 출력
		running_loss += loss.item()
		if (i+1)%2000 == 0: # print every 2000 mini-batches
			print("epoch: {}/{} | step: {}/{} | loss: {:.4f}".format(epoch, num_epochs, i+1, num_batches, running_loss/2000))
			running_loss = 0.0

print("finish Training!")
```
```
모델을 학습시키기 전에,
1. 어떤 loss를 쓸 것인지
2. learning rate은 몇으로 할 것인지
3. optimizer는 무엇으로 할 것인지
4. 학습 횟수는 몇 번으로 할 것인지
를 정해야 하겠죠.  
  
그 다음, 매 epoch, 매 iteration 마다 back propagation을 통해 모델의 파라미터를 업데이트 시켜주는 과정이 필요한데, 아래 다섯 줄의 코드는 공식처럼 외우는 것을 추천드립니다.  
  
optimizer.zero_grad()
pred = model(x)
loss = criterion(pred, x_labels)
loss.backward()
optimizer.step()
```


# Reference
* https://medium.com/@inmoonlight/pytorch%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D%ED%95%98%EA%B8%B0-intro-afd9c67404c3
* https://github.com/inmoonlight/PyTorchTutorial