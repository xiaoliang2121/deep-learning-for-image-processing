import torch
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn
import torch.utils
import torch.utils.data

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    

def get_net():
    net = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    net.apply(init_weights)
    return net

def train(net:nn.Module, train_iter, loss:nn.MSELoss, epochs, lr):
    optim = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            optim.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            optim.step()
        print(f"epoch {epoch+1}, loss: {d2l.evaluate_loss(net, train_iter, loss):f}")
        
def predict(net: nn.Module, data: torch.Tensor):
    net.eval()
    ret = net(data[:, :4])
    # print(f"ret size: {ret.shape}")
    # print(f"data 4 col: {data[:, 4].shape}")
    data[:, 4] = ret.flatten()

def predict_more_step_test(net:nn.Module, x:torch.Tensor, n_train, tau):
    net.eval()
    print(x.shape)
    tt = torch.zeros((x.shape[0]), dtype=torch.float32)
    tt[:n_train+tau] = x[:n_train+tau]
    for i in range(n_train+tau, x.shape[0]):
        tt[i] = net(tt[i-tau:i].reshape((1, -1)))
      
    xx = torch.arange(0, x.shape[0])  
    plt.plot(xx, tt.detach().numpy())
    plt.show()

def main():
    
    # 使用确定模型生成要进行回归的数据
    T = 1000
    time = torch.arange(1, T+1, dtype=torch.float32)
    # print(time)
    x = torch.sin(0.01*time) + torch.normal(0., 0.2, (T,))
    # print(type(x))
    # plt.plot(time.numpy(), x.numpy())
    # plt.plot(time, x)
    # plt.show()
    
    # print(f"cuda: {torch.cuda.is_available()}")
    # print(f"cuda bf16: {torch.cuda.is_bf16_supported()}")
    # d2l.plot([time], [x])
    # plt.show()
    
    # 构造训练数据
    n_train = 600
    tau = 4
    train_data = torch.zeros((T-tau, tau), dtype=torch.float32)
    for i in range(tau):
        train_data[:, i] = x[i:T-tau+i]
    print(train_data.shape)
    # 这里不reshape就结果不对
    # label = x[tau:]
    label = x[tau:].reshape((-1, 1))
    print(label.shape)
    # tt = (train_data, label)
    # print(*tt)
    # print((train_data, label))
    
    data_iter =  d2l.load_array((train_data[:n_train], label[:n_train]), batch_size=16, is_train=True)
    # data_iter =  d2l.load_array((train_data, label), batch_size=16, is_train=True)
    # print(next(iter(data_iter)))
    
    net = get_net()
    loss = nn.MSELoss(reduction='none')
    train(net, data_iter, loss, 5, 0.01)
    
    predict_data = torch.zeros((T-tau, tau+1), dtype=torch.float32)
    predict_data[:, :4] = train_data
    predict(net, predict_data)
    print(predict_data)
    
    predict_more_step_test(net, x, n_train, tau)
    
    # d2l.plot([time[tau:]], [x[tau:], predict_data[:, 4].detach().numpy()])
    # plt.show()
    
if __name__ == "__main__":
    main()