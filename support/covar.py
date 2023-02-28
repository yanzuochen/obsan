class Estimator(object):
    def __init__(self, feature_num, class_num=1, use_cuda=True):
        self.device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
        self.class_num = class_num
        # set it to 1; class-conditional covariance is too costly
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).to(self.device)
        self.Ave = torch.zeros(class_num, feature_num).to(self.device)
        self.Amount = torch.zeros(class_num).to(self.device)

    def calculate(self, features, labels=None):
        # This an incrementally updated covariance.
        # If we don't want class-conditional covar here,
        # simply set `labels` to a vector zeros.
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        if not labels:
            labels = torch.zeros(N, dtype=torch.int64).to(self.device)

        # Expand to prepare for class-conditional covar
        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).to(self.device)
        # Mark the label for each sample, NxC
        # All-one vector if non-class-conditional
        onehot.scatter_(1, labels.view(-1, 1), 1)

        # Each row (last dim) is filled with the same element (0 or 1)
        # All-one array when non-class-conditional
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        # mul is element-wise multiplication so this is just masking.
        # For each sample, preserves only feature rows within enabled classes.
        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        # Sum over the batch dimension to count the times each class appears
        # Each row still has the same element (the count)
        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1  # Avoid division by zero

        # Average of each feature in each class
        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),  # CxAxN
            var_temp.permute(1, 0, 2)  # CxNxA
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        # For each class, make a matrix filled with the same element (the count)
        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        new_CoVariance = (self.CoVariance.mul(1 - weight_CV) +
                          var_temp.mul(weight_CV)).detach() + additional_CV.detach()

        new_Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        new_Amount = self.Amount + onehot.sum(0)

        return {
            'Ave': new_Ave, 
            'CoVariance': new_CoVariance,
            'Amount': new_Amount
        }

if __name__ == '__main__':
    
    train_loader = [] # stores training data
    test_loader = [] # stores test data

    # 1. for each layer, first initialize an Estimator
    covar_dict = {}
    for layer in model:
        covar_dict[layer.name] = Estimator(layer.feature_num, 1)

    # 2. during training, update the covariance

    for (data, label) in train_loader:
        layer_output_dict = model(data) #
        for (layer_name, layer_output) in layer_output_dict.items():
            temp = covar_dict[layer_name].calculate(layer_output)
            covar_dict[layer_name].update(temp)

    # 3. during testing, detect OOD as follows:
    for (data, label) in test_loader:
        layer_output_dict = model(data) #
        for (layer_name, layer_output) in layer_output_dict.items():
            temp = covar_dict[layer_name].calculate(layer_output)
            # compare temp.covar with covar_dict[layer_name].covar
            # you can simply use sum(abs(temp.covar - covar_dict[layer_name].covar))
