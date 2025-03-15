import torch
from tqdm import tqdm
from prettytable import PrettyTable
def compute_cos(loader,
                target_model_list,
                model,
                batch_size):
        model.zero_grad()
        model.eval()
        for target_model in target_model_list:
            target_model.eval()
        
        sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        train_criterion = torch.nn.CrossEntropyLoss()
        iterator = tqdm(enumerate(loader), total=len(loader))
        cos_map = torch.zeros(len(target_model_list)).cuda()
        for i, (input, target) in iterator:
            target_grad = []
            for idx, target_model in enumerate(target_model_list):
                inp = input.cuda(non_blocking=True).requires_grad_(True)
                target = target.cuda(non_blocking=True)
                output = target_model(inp)
                loss = train_criterion(output, target)
                grad = torch.autograd.grad(loss,
                                           inp,
                                           retain_graph=False,
                                           create_graph=False)[0]
                target_grad.append(torch.flatten(grad,1))
            inp = input.cuda(non_blocking=True).requires_grad_(True)
            target = target.cuda(non_blocking=True)
            output = model(inp)
            loss = train_criterion(output, target)
            model_grad = torch.autograd.grad(loss,
                                        inp,
                                        retain_graph=False,
                                        create_graph=False)[0]
            model_grad = torch.flatten(model_grad,1)

            for j in range(len(target_model_list)):
                cos_map[j] += torch.sum(sim_func(target_grad[j],model_grad))/batch_size
        cos_map = cos_map/len(loader)
        results = PrettyTable(field_names=["Model name", "cosine_sim"])  
        for i in range(len(target_model_list)):
            results.add_row([
                f"model{i}",
                f"{cos_map[i]:.8f}",
            ])
        print(results)
        return cos_map