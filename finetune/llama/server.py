import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from itertools import combinations
import numpy as np
import logging
from client import Client



# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 确保日志目录存在
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# 设置日志记录
logging.basicConfig(filename=os.path.join(log_dir, 'LoRA.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


class FederatedServer:
    
    def __init__(self, clients):
        self.clients = clients                 
        self.num_clients=len(clients)
    
    #初始时为了使所有节点的LoRA参数一致
    def aggregate_lora_A(self):
        self.Avg_lora_A_params=self.clients[0].get_lora_A()
        for i in range(self.num_clients):
            self.clients[i].set_trainable_parameters(self.Avg_lora_A_params)
              
    def aggregate_dfl(self, A):
        self.new_params = []
        
        # 初始化每个客户端的参数为零
        for i in range(self.num_clients):
            client_params = self.clients[i].get_lora_parameters()
            zero_params = {name: (torch.zeros_like(param[0]), param[1]) for name, param in client_params.items()}
            self.new_params.append(zero_params)
        
        # 聚合操作
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                client_params = self.clients[j].get_lora_parameters()
                for name, (param, requires_grad) in client_params.items():
                    self.new_params[i][name] = (self.new_params[i][name][0] + param * A[i][j], requires_grad)
                    
        for i in range(self.num_clients):
            self.clients[i].set_trainable_parameters(self.new_params[i])
                               

    # 提取并相乘lora参数
    def extract_and_multiply_lora_params(self,param_group):
        result = {}
        for param_name, (param, _) in param_group.items():
            if 'lora_B.default.weight' in param_name:
                prefix = param_name.split('lora_B.default.weight')[0]
                lora_A_name = prefix + 'lora_A.default.weight'
                lora_B_name = prefix + 'lora_B.default.weight'
                
                if lora_A_name in param_group and lora_B_name in param_group:
                    lora_A = param_group[lora_A_name][0]
                    lora_B = param_group[lora_B_name][0]
                    
                    product = torch.matmul(lora_B, lora_A)
                    result[prefix + 'product'] = product
                    
        return result


    # 计算所有参数组两两之间的差异并求平均值
    def calculate_lora_products_and_avg_diff(self,param_groups):
        if len(param_groups) < 2:
            raise ValueError("There should be at least two sets of parameters to calculate differences.")
        
        total_diff_sum = 0.0
        num_pairs = 0

        # 生成所有参数组的两两组合
        for i, j in combinations(range(len(param_groups)), 2):
            product_1 = self.extract_and_multiply_lora_params(param_groups[i])
            product_2 = self.extract_and_multiply_lora_params(param_groups[j])
            pair_diff_sum=0.0
            # 计算 lora_A 和 lora_B 的乘积差异
            for key in product_1.keys():
                diff = product_1[key] - product_2[key]
                #pair_diff_sum += torch.sum(diff).item()
                pair_diff_sum += torch.norm(diff).item()

            # 计算普通参数的差异
            for param_name in param_groups[i].keys():
                if 'lora_A.default.weight' in param_name or 'lora_B.default.weight' in param_name:
                    continue  # 跳过 lora_A 和 lora_B，因为已经处理过
                diff = param_groups[i][param_name][0] - param_groups[j][param_name][0]
                #pair_diff_sum += torch.sum(diff).item()
                pair_diff_sum += torch.norm(diff).item()
                
            
            total_diff_sum += pair_diff_sum

            num_pairs += 1

        # 计算平均差异
        average_diff = total_diff_sum / num_pairs if num_pairs > 0 else 0.0
        return average_diff

def get_A(name): 
    if name=="du14":
        A = np.array([
    [0, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0]
    ])
    if name=="link3":
        A = np.array([
    [0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0]
    ])
    elif name=="link4":
        A= np.array([
    [0, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0]
    ])
    elif name=="link5":
        A= np.array([
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 1, 1, 0]
    ])
    elif name=="link6":
        A= np.array([
    [0, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 0],
    [1, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0]
    ])
    elif name=="link7":
        A= np.array([
    [0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0]
    ])
    
    degree = np.sum(A, axis=1)
    
    # 初始化权重矩阵W
    W = np.zeros_like(A, dtype=float)

    # 计算权重矩阵
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                W[i, j] = 1 / (max(degree[i], degree[j]) + 1)
            elif i != j:
                W[i, j] = 0

    # 处理对角线元素
    for i in range(A.shape[0]):
        W[i, i] = 1 - np.sum(W[i, np.arange(A.shape[0]) != i])
    
        
    return W           
                     
            
if __name__=="__main__":
    model_checkpoint = '/home/ubuntu/smyin/models/Llama-3.2-1B'
    dataset_path_template = "/home/ubuntu/smyin/dataset/decentrilized_dataset/gsm8k/client_{}"

    clients = []
    num_clients = 7
    type = "LoRA"
    name="link3"
    batch_size=2

    

    for i in range(num_clients):
        dataset_path = dataset_path_template.format(i+1)
        #device = f"cuda:{i % 4}"  # 循环分配到 4 张 GPU
        client = Client(model_path=model_checkpoint, data_path=dataset_path, type=type,batch_size=batch_size)
        clients.append(client)

    server = FederatedServer(clients)

    server.aggregate_lora_A()
  
    num_rounds = 3
    diff = []
    
    A=get_A(name)
    
    server.aggregate_dfl(A)
    a = server.calculate_lora_products_and_avg_diff(server.new_params)
    diff.append(a)

    for round in range(num_rounds):
        logging.info(f"Round {round+1}/{num_rounds}")
        for client in server.clients:
            loss = client.train_one_epoch()
            logging.info(f"Client {server.clients.index(client) + 1} - Loss: {loss}")

        server.aggregate_dfl(A)
        a = server.calculate_lora_products_and_avg_diff(server.new_params)
        diff.append(a)
        logging.info(f"Round {round+1} - Parameter Difference: {a}")
        

        for i in range(server.num_clients):
            server.clients[i].trainable_parameters_nums()
        logging.info(f"Parameter Difference record: {diff}")
