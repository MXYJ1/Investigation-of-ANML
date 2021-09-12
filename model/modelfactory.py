class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, in_channels=6, num_actions=6, width=300):

        if "Sin" == dataset:
            if model_type=="old":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size * 3, hidden_size]),
                    ('relu', [True]),
                    ('rep', []),
                    ('linear', [hidden_size, hidden_size * 3]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [num_actions, hidden_size])
                ]
            elif model_type=="linear":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size * 5, hidden_size]),
                    ('relu', [True]),
                    ('rep', []),
                    ('linear', [num_actions, hidden_size * 5])
                ]

            elif model_type=="non-linear":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size * 5, hidden_size]),
                    ('relu', [True]),
                    ('rep', []),
                    ('linear', [hidden_size, hidden_size * 5]),
                    ('relu', [True]),
                    ('linear', [num_actions, hidden_size])

                ]

        elif dataset == "omniglot":
            
            if model_type == "Neuromodulation":

                nm_channels = 112
                channels = 256
                size_of_representation = 2304
                size_of_interpreter = 1008

                return [
                    
                    # =============== Separate network neuromodulation =======================

                    ('conv1_nm', [nm_channels, 3, 3, 3, 1, 0]), # Model 1 and Model 4 variation 1
                    #('conv1_nm', [nm_channels, 3, 10, 10, 1, 0]), # Model 4 variation 2
                    #('conv1_nm', [nm_channels, 3, 18, 18, 1, 0]), # Model 4
                    #('conv1_nm', [nm_channels, 3, 26, 26, 1, 0]), # Model 5
                    
                    #('conv1_nm', [nm_channels, 3, 10, 10, 1, 0]), # Modify 1
                    #('conv1_nm', [nm_channels, 3, 8, 8, 1, 0]), # Modify 2
                    #('conv1_nm', [nm_channels, 3, 7, 7, 1, 0]), # Modify 3
                    
                    ('bn1_nm', [nm_channels]), 
                    
                    ('conv2_nm', [nm_channels, nm_channels, 3, 3, 1, 0]), # Model 1 and Model 4
                    #('conv2_nm', [nm_channels, nm_channels, 11, 11, 1, 0]), # Model 4 variation 1
                    #('conv2_nm', [nm_channels, nm_channels, 7, 7, 1, 0]), # Model 4 variation 2
                    
                    #('conv2_nm', [nm_channels, nm_channels, 9, 9, 1, 0]), # modify 1
                    #('conv2_nm', [nm_channels, nm_channels, 7, 7, 1, 0]), # modify 2
                    #('conv2_nm', [nm_channels, nm_channels, 6, 6, 1, 0]), # modify 3
                    
                    ('bn2_nm', [nm_channels]),
                    
                    
                    ('conv3_nm', [nm_channels, nm_channels, 3, 3, 1, 0]), # Model 1
                    #('conv3_nm', [nm_channels, nm_channels, 9, 9, 1, 0]), # modify 1
                    #('conv3_nm', [nm_channels, nm_channels, 7, 7, 1, 0]), # modify 2
                    #('conv3_nm', [nm_channels, nm_channels, 6, 6, 1, 0]), # modify 3
                    
                    ('bn3_nm', [nm_channels]),
                    
                    
                    #('conv4_nm', [nm_channels, nm_channels, 7, 7, 1, 0]), # modify 2
                    #('conv4_nm', [nm_channels, nm_channels, 6, 6, 1, 0]), # modify 3
                    
                    #('conv5_nm', [nm_channels, nm_channels, 5, 5, 1, 0]), # modify 3
                    
                    ('nm_to_fc', [size_of_representation, size_of_interpreter]),
                    #('scale',[]), # when scaling sigmoid 
                    #('scale',[1, size_of_representation]), # when scaling elements of gating vector

                    # =============== Prediction network ===============================

                    ('conv1', [channels, 3, 3, 3, 1, 0]),
                    ('bn1', [channels]),
                    ('conv2', [channels, channels, 3, 3, 1, 0]),
                    ('bn2', [channels]),
                    ('conv3', [channels, channels, 3, 3, 1, 0]),
                    ('bn3', [channels]),
                    ('fc', [1000, size_of_representation]),
                ]


            elif model_type == 'OML':
                channels = 256
                size_of_representation = 2304
                return [
                    ('conv2d', [channels, 3, 3, 3, 1, 0]),
                    #('bn', [channels]), 
                    ('relu', [True]),
                    #('max_pool2d', [2, 2, 0]),
                    ('conv2d', [channels, channels, 3, 3, 1, 0]),
                    #('bn', [channels]),
                    ('relu', [True]),
                    #('max_pool2d', [2, 2, 0]),
                    ('conv2d', [channels, channels, 3, 3, 1, 0]),
                    #('bn', [channels]),
                    ('relu', [True]),
                    #('max_pool2d', [2, 2, 0]),
                    ('conv2d', [channels, channels, 3, 3, 2, 0]),
                    ('relu', [True]),
                    ('conv2d', [channels, channels, 3, 3, 1, 0]),
                    ('relu', [True]),
                    ('conv2d', [channels, channels, 3, 3, 2, 0]),
                    ('relu', [True]),
                    ('flatten', []),
                    ('rep', []),
                    
                    #('linear', [1000, size_of_representation]),
                    
                    ('linear', [1024, 2304]),
                    ('relu', [True]),
                    ('linear', [1000, 1024]),
                ]

        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
