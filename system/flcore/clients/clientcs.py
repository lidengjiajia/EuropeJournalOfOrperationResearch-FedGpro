import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientCS(Client):
    """
    FedCS Client: Federated Crow Search-Based Dynamic Aggregation
    
    Client-side implementation for FedCS algorithm.
    Each client acts as a "crow" in the crow search algorithm optimization framework.
    
    Key Features:
    - Standard local training (same as FedAvg)
    - Returns validation accuracy for server-side ranking
    - Receives CSA-optimized parameters from server
    
    Category: Traditional FL (tFL) - Advanced Aggregation
    
    Reference: "FedCS: Federated Learning with Crow Search-Based Dynamic Aggregation"
    Based on: Crow Search Algorithm (Askarzadeh, 2016)
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Client's validation accuracy (used by server for ranking)
        self.validation_accuracy = 0.0
        
        # Track if this client is the best or second-best
        self.is_best = False
        self.is_second_best = False

    def train(self):
        """
        Standard local training procedure.
        Same as FedAvg - the CSA optimization happens on server side.
        """
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                output = self.model(x)
                loss = self.loss(output, y)
                
                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Client {self.id} detected NaN/Inf loss, enabling gradient clipping")
                    self.enable_grad_clip = True
                    self.nan_detected_count += 1
                    continue  # Skip this batch
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Only clip gradients if NaN was previously detected
                if self.enable_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def get_validation_accuracy(self):
        """
        Compute validation accuracy for server-side ranking.
        Uses test data as validation set (common practice in FL).
        
        Returns:
            float: Validation accuracy in [0, 1]
        """
        testloader = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        accuracy = test_acc / test_num if test_num > 0 else 0.0
        self.validation_accuracy = accuracy
        
        return accuracy
